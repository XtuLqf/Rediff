import torch
import ast
import io
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys
import subprocess

import numpy as np
import pytest

from datasets.image_util import DATA_LOADER
from relation.topology import sdga_pair_losses
from relation.timestep_schedule import mixed_relation_groups

from diagnostics.relation_metrics import class_relation_loss, instance_relation_loss
from relation.timestep_schedule import (
    sample_relation_group_timesteps,
    sample_relation_weights,
)
from relation.topology import time_aware_pair_losses
from relation.vsra import TimeAwareVSRA


def test_grouped_timesteps_keep_classes_together_and_balance_loads():
    torch.manual_seed(7)
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7])

    timesteps = sample_relation_group_timesteps(labels, n_timesteps=4)

    for label in labels.unique():
        assert timesteps[labels == label].unique().numel() == 1
    loads = torch.bincount(timesteps, minlength=4)
    largest_class = max(int((labels == label).sum()) for label in labels.unique())
    assert int(loads.max() - loads.min()) <= largest_class


def test_time_aware_pairs_exclude_cross_timestep_relations():
    student = torch.randn(4, 5)
    semantic = torch.randn(4, 3)
    contrastive = torch.randn(4, 6)
    labels = torch.tensor([0, 0, 1, 1])
    timesteps = torch.tensor([0, 0, 1, 1])
    weights = torch.ones(4)

    losses = time_aware_pair_losses(
        student,
        semantic,
        contrastive,
        labels,
        timesteps,
        weights,
        weights,
        topology_norm="timestep",
    )

    assert losses["class_pairs"].item() == 0
    assert losses["instance_pairs"].item() == 4
    assert losses["class"].item() == 0.0


def test_diagnostics_match_training_topology_definition():
    torch.manual_seed(11)
    student = torch.randn(6, 5)
    semantic = torch.randn(6, 3)
    contrastive = torch.randn(6, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    timesteps = torch.zeros(6, dtype=torch.long)
    weights = torch.ones(6)

    losses = time_aware_pair_losses(
        student,
        semantic,
        contrastive,
        labels,
        timesteps,
        weights,
        weights,
    )

    assert torch.allclose(losses["class"], class_relation_loss(student, semantic, labels))
    assert torch.allclose(
        losses["instance"],
        instance_relation_loss(student, contrastive, labels),
    )


def test_each_timestep_topology_is_normalized_independently():
    student = torch.tensor(
        [[0.0], [1.0], [3.0], [4.0], [100.0], [110.0], [130.0], [140.0]]
    )
    teacher = torch.tensor(
        [[0.0], [1.0], [3.0], [4.0], [10.0], [11.0], [13.0], [14.0]]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    timesteps = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    weights = torch.ones(8)

    losses = time_aware_pair_losses(
        student,
        teacher,
        teacher,
        labels,
        timesteps,
        weights,
        weights,
        topology_norm="timestep",
    )

    assert torch.allclose(losses["class"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(losses["instance"], torch.tensor(0.0), atol=1e-6)

    global_losses = time_aware_pair_losses(
        student,
        teacher,
        teacher,
        labels,
        timesteps,
        weights,
        weights,
        topology_norm="global",
    )
    assert global_losses["class"].item() > 0.0
    assert global_losses["instance"].item() > 0.0


def test_diffusion_reliability_decays_without_high_noise_amplification():
    timesteps = torch.arange(4)
    signal_retention = torch.tensor([0.64, 0.16, 0.04, 0.01])

    class_weights, instance_weights = sample_relation_weights(
        timesteps,
        n_timesteps=4,
        mode="diffusion_reliability",
        strength=0.5,
        signal_retention=signal_retention,
        reliability_floor=0.5,
    )

    assert torch.all(class_weights[:-1] >= class_weights[1:])
    assert torch.all(instance_weights[:-1] >= instance_weights[1:])
    assert torch.all(class_weights >= instance_weights)
    assert torch.all(class_weights <= 1.0)
    assert torch.all(instance_weights <= 1.0)
    assert torch.all(class_weights >= 0.5)
    assert torch.all(instance_weights >= 0.5)

    fixed_class, fixed_instance = sample_relation_weights(
        timesteps,
        n_timesteps=4,
        mode="diffusion_reliability",
        strength=0.0,
        signal_retention=signal_retention,
        reliability_floor=0.5,
    )
    assert torch.equal(fixed_class, torch.ones(4))
    assert torch.equal(fixed_instance, torch.ones(4))


def test_static_and_time_aware_losses_use_a_convex_budget():
    torch.manual_seed(13)
    generated = torch.randn(6, 4)
    semantic = torch.randn(6, 3)
    contrastive = torch.randn(6, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    timesteps = torch.zeros(6, dtype=torch.long)

    static = TimeAwareVSRA(
        n_timesteps=4,
        visual_dim=4,
        contrastive_dim=4,
        projection_dim=3,
        angle_ratio=0.0,
        time_pair_weight=0.0,
    )
    static_losses = static(generated, semantic, contrastive, labels, timesteps)
    assert torch.allclose(static_losses["total"], static_losses["legacy_total"])
    assert static_losses["pair_total"].item() == 0.0

    mixed = TimeAwareVSRA(
        n_timesteps=4,
        visual_dim=4,
        contrastive_dim=4,
        projection_dim=3,
        angle_ratio=0.0,
        time_pair_weight=0.25,
    )
    mixed_losses = mixed(generated, semantic, contrastive, labels, timesteps)
    expected = 0.75 * mixed_losses["legacy_total"] + 0.25 * mixed_losses["pair_total"]
    assert torch.allclose(mixed_losses["total"], expected)

    time_only = TimeAwareVSRA(
        n_timesteps=4,
        visual_dim=4,
        contrastive_dim=4,
        projection_dim=3,
        angle_ratio=0.0,
        time_pair_weight=1.0,
    )
    time_losses = time_only(generated, semantic, contrastive, labels, timesteps)
    assert time_losses["legacy_total"].item() == 0.0
    assert torch.allclose(time_losses["total"], time_losses["pair_total"])


@pytest.mark.parametrize('sizes', [(6, 6), (6, 9)])
def test_sdga_equal_state_reduction_and_legacy_pair_reduction(sizes):
    torch.manual_seed(23)
    labels = torch.arange(sum(sizes) // 3).repeat_interleave(3)
    groups = torch.cat([torch.full((n,), t, dtype=torch.long) for t, n in enumerate(sizes)])
    student, semantic, con = [torch.randn(sum(sizes), 5) for _ in range(3)]
    blocks = sdga_pair_losses(student, semantic, con, labels, groups, 2)
    old = time_aware_pair_losses(student, semantic, con, labels, groups,
                                torch.ones(sum(sizes)), torch.ones(sum(sizes)), topology_norm='timestep')
    for name in ('class', 'instance'):
        # Independently evaluate each state's data with the existing diagnostic.
        metric = class_relation_loss if name == 'class' else instance_relation_loss
        teacher = semantic if name == 'class' else con
        expected = torch.stack([metric(student[groups == t], teacher[groups == t], labels[groups == t])
                                for t in range(2)])
        assert torch.allclose(blocks[name], expected.mean(), atol=1e-6)
        counts = blocks[name + '_pair_count']
        assert torch.allclose(old[name], (expected * counts).sum() / counts.sum(), atol=1e-6)
        if sizes[0] == sizes[1]:
            assert torch.allclose(blocks[name], old[name], atol=1e-6)
        else:
            assert not torch.allclose(blocks[name], old[name], atol=1e-6)


def test_sdga_empty_blocks_and_single_edge_have_safe_gradients():
    student = torch.randn(2, 4, requires_grad=True)
    teacher = torch.randn(2, 4, requires_grad=True)
    # Two isolated states have no relation edges of either granularity.
    empty = sdga_pair_losses(student, teacher, teacher, torch.tensor([0, 1]), torch.tensor([0, 1]), 3)
    assert empty['class'].item() == empty['instance'].item() == 0
    (empty['class'] + empty['instance']).backward()
    assert torch.equal(student.grad, torch.zeros_like(student))
    assert teacher.grad is None
    for labels, name in [(torch.tensor([0, 1]), 'class'), (torch.tensor([0, 0]), 'instance')]:
        single = sdga_pair_losses(student, teacher, teacher, labels, torch.zeros(2, dtype=torch.long), 3)
        assert single[name + '_degenerate'].tolist() == [True, False, False]
        assert single[name + '_pair_count'].tolist() == [2, 0, 0]
        assert single[name].item() == pytest.approx(0, abs=1e-6)


def test_sdga_skips_empty_states_and_differentiates_student_normalization():
    torch.manual_seed(37)
    student = torch.randn(6, 4, dtype=torch.double, requires_grad=True)
    teacher = torch.randn(6, 4, dtype=torch.double, requires_grad=True)
    labels = torch.arange(3).repeat_interleave(2)
    groups = torch.zeros(6, dtype=torch.long)
    result = sdga_pair_losses(student, teacher, teacher, labels, groups, 4)
    assert torch.allclose(result['class'], class_relation_loss(student, teacher, labels))
    assert torch.allclose(result['instance'], instance_relation_loss(student, teacher, labels))
    assert torch.autograd.gradcheck(
        lambda x: sdga_pair_losses(x, teacher, teacher, labels, groups, 4)['class'], (student,),
    )
    (result['class'] + result['instance']).backward()
    assert student.grad.norm() > 0 and torch.isfinite(student.grad).all()
    assert teacher.grad is None


@pytest.mark.parametrize('objective,eta', [('legacy', 0), ('legacy', 1), ('sdga', 1)])
def test_generator_coefficients_preserve_calibration_and_frozen_projector_gradients(objective, eta):
    torch.manual_seed(43)
    common = dict(n_timesteps=2, visual_dim=4, contrastive_dim=4, projection_dim=3,
                  objective=objective, time_pair_weight=eta, time_mode='fixed', time_strength=0)
    enabled = TimeAwareVSRA(**common)
    disabled = TimeAwareVSRA(**common, generator_class_weight=0, generator_instance_weight=0)
    disabled.load_state_dict(enabled.state_dict())
    x = torch.randn(12, 4, requires_grad=True)
    semantic, con = torch.randn(12, 3), torch.randn(12, 4, requires_grad=True)
    labels = torch.arange(6).repeat_interleave(2)
    groups = torch.arange(2).repeat_interleave(6)
    assert torch.equal(enabled.calibration_losses(x, semantic, con)['total'],
                       disabled.calibration_losses(x, semantic, con)['total'])
    assert disabled(x, semantic, con, labels, groups)['total'].item() == 0
    for parameter in enabled.parameters():
        parameter.requires_grad_(False)
    enabled(x, semantic, con, labels, groups)['total'].backward()
    assert x.grad.norm() > 0 and con.grad is None
    assert all(p.grad is None for p in enabled.parameters())


def synthetic_data(counts, visual_dim=4):
    data = DATA_LOADER.__new__(DATA_LOADER)
    data.train_label = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    data.ntrain = data.train_label.numel()
    # First visual coordinate uniquely identifies the sampled training row.
    data.train_feature = torch.randn(data.ntrain, visual_dim)
    data.train_feature[:, 0] = torch.arange(data.ntrain)
    data.train_paco = torch.randn(data.ntrain, 2048)
    data.attribute = torch.randn(len(counts), 4)
    data.seenclasses = torch.arange(len(counts))
    data.unseenclasses = torch.empty(0, dtype=torch.long)
    return data


def test_pk_sampling_eligibility_uniqueness_and_rng_resume():
    data = synthetic_data([7] * 20 + [3])
    assert data.prepare_pk_sampler(16, 4) == {'eligible_classes': 20, 'excluded_classes': 1}
    rng = torch.Generator().manual_seed(71)
    state = rng.get_state()
    original_global_state = torch.get_rng_state()
    first = data.next_seen_pk_batch(rng)
    assert torch.equal(original_global_state, torch.get_rng_state())
    assert first[0][:, 0].unique().numel() == 64
    assert torch.equal(first[-1].unique(return_counts=True)[1], torch.full((16,), 4))
    rng.set_state(state)
    assert all(torch.equal(a, b) for a, b in zip(first, data.next_seen_pk_batch(rng)))
    with pytest.raises(ValueError, match='only 20 available'):
        data.prepare_pk_sampler(21, 4)


@pytest.mark.parametrize('n_states,classes_per_state', [(4, 4), (3, 4), (4, 3)])
def test_mixed_groups_preserve_pair_budget_and_generator_states(n_states, classes_per_state):
    labels = torch.arange(n_states * classes_per_state).repeat_interleave(4)
    times = sample_relation_group_timesteps(labels, n_states, torch.Generator().manual_seed(81))
    original = times.clone()
    global_state = torch.get_rng_state()
    groups = mixed_relation_groups(labels, times, n_states, torch.Generator().manual_seed(91))
    assert torch.equal(times, original) and torch.equal(global_state, torch.get_rng_state())
    for c in labels.unique():
        assert groups[labels == c].unique().numel() == 1
    for t in range(n_states):
        assert times[groups == t].unique().numel() >= 2
    features = torch.randn(labels.numel(), 4)
    matched = sdga_pair_losses(features, features, features, labels, times, n_states)
    mixed = sdga_pair_losses(features, features, features, labels, groups, n_states)
    for key in ('class_pair_count', 'instance_pair_count', 'samples_per_block', 'classes_per_block'):
        assert torch.equal(matched[key], mixed[key])
    if n_states == classes_per_state == 4:
        assert mixed['class_pair_count'].tolist() == [192] * 4
        assert mixed['instance_pair_count'].tolist() == [48] * 4


def parse_options(monkeypatch, *arguments):
    monkeypatch.setattr(sys, 'argv', ['config_zerodiff.py', *arguments])
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / 'config_zerodiff.py'))['opt']


@pytest.mark.parametrize('arguments', [
    ['--rel_objective', 'sdga'],
    ['--g_batch_mode', 'pk', '--g_pk_samples', '3'],
    ['--g_batch_mode', 'pk', '--g_pk_classes', '8', '--g_pk_samples', '8'],
    ['--rel_pair_grouping', 'mixed'],
    ['--rel_generator_class_weight', 'nan'],
])
def test_invalid_experiment_settings_fail_early(monkeypatch, arguments):
    with pytest.raises(SystemExit):
        parse_options(monkeypatch, *arguments)


@pytest.mark.parametrize('dataset', ['awa2', 'cub', 'sun'])
def test_launcher_accepts_explicit_drg_outside_default_directory(monkeypatch, tmp_path, dataset):
    checkpoint = tmp_path / 'custom_drg.tar'
    checkpoint.touch()
    launcher = Path(__file__).resolve().parents[1] / 'scripts' / f'run_{dataset}_zerodiff_DFG_train.py'
    monkeypatch.setattr(sys, 'argv', [str(launcher), '--netR_model_path', str(checkpoint), '--nepoch', '1'])
    calls = []
    monkeypatch.setattr('subprocess.run', lambda command, **kwargs: calls.append(command))
    runpy.run_path(str(launcher))
    assert calls[0][-2:] == ['--netR_model_path', str(checkpoint)]
    assert calls[0][-4:-2] == ['--nepoch', '1']


@pytest.mark.parametrize('experiment,grouping,weight', [('S0', 'matched', 0), ('S1', 'matched', 1), ('S2', 'mixed', 1)])
def test_awa2_short_experiments_keep_settings_and_skip_existing_runs(monkeypatch, tmp_path, experiment, grouping, weight):
    root = Path(__file__).resolve().parents[1]
    checkpoint = tmp_path / 'drg.tar'
    checkpoint.touch()
    run_name = {'S0': 's0_control', 'S1': 's1_matched', 'S2': 's2_mixed'}[experiment]
    base = root / 'out' / 'ds_reg' / 'AWA2' / f'{run_name}_seed9182'
    original_exists = Path.exists
    monkeypatch.setattr(Path, 'exists', lambda path: path in (base, base.with_name(base.name + '_run2')) or original_exists(path))
    monkeypatch.setattr(sys, 'argv', ['launcher', '--experiment', experiment, '--netR_model_path', str(checkpoint)])
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs: calls.append(command))
    runpy.run_path(str(root / 'scripts' / 'run_awa2_zerodiff_DFG_train.py'))
    options = parse_options(monkeypatch, *calls[0][2:])
    assert options.run_dir == str(base.with_name(base.name + '_run3'))
    assert options.rel_objective == 'sdga' and options.gamma_rel == 1
    assert options.rel_pair_grouping == grouping
    assert options.rel_generator_class_weight == options.rel_generator_instance_weight == weight
    assert options.rel_class_weight == options.rel_instance_weight == options.rel_teacher_anchor_weight == 1
    assert options.g_batch_mode == 'pk' and options.g_timestep_policy == 'class_group'
    assert (options.g_pk_classes, options.g_pk_samples, options.batch_size, options.n_T) == (16, 4, 64, 4)
    assert (options.rel_time_mode, options.rel_time_strength, options.rel_time_pair_weight) == ('fixed', 0, 1)
    assert (options.nepoch, options.eval_interval, options.syn_num, options.manualSeed) == (300, 5, 5400, 9182)


def test_awa2_short_resume_uses_checkpoint_directory_and_preserves_exit_code(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    drg = tmp_path / 'drg.tar'
    drg.touch()
    resume = tmp_path / 'prior_run' / 'dfg_training_last.tar'
    monkeypatch.setattr(sys, 'argv', ['launcher', '--experiment', 'S1', '--netR_model_path', str(drg),
                                     '--resume_training', str(resume), '--nepoch', '400'])
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(7, command)
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(root / 'scripts' / 'run_awa2_zerodiff_DFG_train.py'))
    assert caught.value.code == 7
    options = parse_options(monkeypatch, *calls[0][2:])
    assert options.run_dir == str(resume.parent)
    assert options.nepoch == 400 and options.resume_training == str(resume)


def training_namespace(options):
    """Load real trainer definitions without executing its dataset/GPU entrypoint."""
    import zerodiff_tools
    path = Path(__file__).resolve().parents[1] / 'zerodiff_DFG_train.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    definitions = ast.Module(body=[node for node in tree.body
                                  if isinstance(node, (ast.FunctionDef, ast.ClassDef))], type_ignores=[])
    scope = dict(torch=torch, optim=torch.optim, np=np, random=random, os=os, json=json,
                 opt=options, zerodiff_tools=zerodiff_tools, TimeAwareVSRA=TimeAwareVSRA,
                 sample_relation_group_timesteps=sample_relation_group_timesteps,
                 mixed_relation_groups=mixed_relation_groups, logger=io.StringIO(),
                 new_relation_behavior=True, BEST_STATE_NAMES=(),
                 relation_run_config={'objective': 'sdga'}, training_run_config={'seed': 103})
    exec(compile(definitions, str(path), 'exec'), scope)
    scope['relation_run_config'] = scope['normalized_relation_config'](scope['relation_run_config'])
    return scope


@pytest.mark.parametrize('grouping', ['matched', 'mixed'])
def test_real_training_step_and_full_resume_are_identical(monkeypatch, tmp_path, grouping):
    options = parse_options(monkeypatch, '--dataset', 'AWA2', '--gamma_rel', '1',
                            '--rel_objective', 'sdga', '--rel_time_mode', 'fixed', '--rel_time_strength', '0',
                            '--g_batch_mode', 'pk', '--g_timestep_policy', 'class_group',
                            '--rel_pair_grouping', grouping, '--manualSeed', '103',
                            '--resSize', '8', '--attSize', '4', '--noiseSize', '4', '--dim_t', '4',
                            '--ndh', '16', '--encoder_layer_sizes', '8', '16',
                            '--decoder_layer_sizes', '16', '8', '--rel_proj_dim', '8')
    options.cuda = False
    options.critic_iter = 1
    torch.manual_seed(103)
    torch.set_num_threads(1)
    data = synthetic_data([8] * 20, visual_dim=8)
    data.train_feature = torch.sigmoid(data.train_feature)
    scope = training_namespace(options)
    scope.update(data=data, input_res=torch.empty(64, 8), input_con=torch.empty(64, 2048),
                 input_att=torch.empty(64, 4), input_label=torch.empty(64, dtype=torch.long),
                 sampleTestSeen=lambda: data.next_seen_batch(64)[:3])
    drg = scope['zerodiff_tools'].DRG_Generator(options)
    drg_path = tmp_path / 'drg.tar'
    torch.save({'state_dict_G_con': drg.state_dict()}, drg_path)

    def new_model():
        return scope['ZERODIFF'](data, 4, (0.1, 20), data.seenclasses, data.unseenclasses,
                                 data.attribute, str(drg_path), device='cpu')

    model = new_model()
    calibration_inputs, generator_inputs = [], []
    model.time_aware_vsra.visual_projector.register_forward_pre_hook(
        lambda module, args: calibration_inputs.append(args[0].detach().clone()))
    model.netE.register_forward_pre_hook(lambda module, args: generator_inputs.append(args[0].detach().clone()))
    model()
    # D's batch is calibrated, then one newly sampled batch is used by E/G.
    assert torch.equal(calibration_inputs[0], generator_inputs[0])
    assert not torch.equal(generator_inputs[0], generator_inputs[1])
    summary = model.relation_block_summary(0)
    assert summary['class_pair_count'] == [192] * 4
    assert summary['instance_pair_count'] == [48] * 4
    assert summary['class_valid'] == summary['instance_valid'] == [1] * 4
    assert all(not tensor.requires_grad for tensor in model.relation_block_sums.values())
    model.init_recorder()
    checkpoint = scope['save_training_state'](model, str(tmp_path / 'dfg'), 0)
    expected_losses = [x.detach().clone() for x in model()]
    expected_weights = {key: value.clone() for key, value in model.state_dict().items()}
    restored = new_model()
    assert scope['load_training_state'](restored, checkpoint) == 1
    resumed_losses = restored()
    for expected, resumed in zip(expected_losses, resumed_losses):
        assert torch.equal(expected, resumed.detach())
    for key, expected in expected_weights.items():
        assert torch.equal(expected, restored.state_dict()[key]), key
    scope['training_run_config']['seed'] = 999
    with pytest.raises(ValueError, match='training configuration differs'):
        scope['load_training_state'](restored, checkpoint)


def test_old_checkpoint_defaults_do_not_adopt_sdga_settings(monkeypatch):
    scope = training_namespace(parse_options(monkeypatch))
    normalized = scope['normalized_relation_config']({'class_weight': 2, 'instance_weight': 3})
    assert normalized['objective'] == 'legacy'
    assert normalized['generator_class_weight'] == 2 and normalized['generator_instance_weight'] == 3
    assert normalized['g_timestep_policy'] == 'auto' and normalized['g_batch_mode'] == 'random'


def trainer_startup_code():
    path = Path(__file__).resolve().parents[1] / 'zerodiff_DFG_train.py'
    nodes = ast.parse(path.read_text(encoding='utf-8')).body
    start = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == 'ensure_cuda_ready')
    end = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'data' for target in node.targets))
    return compile(ast.Module(body=nodes[start:end], type_ignores=[]), str(path), 'exec')


def test_cuda_failure_stops_before_creating_outputs(monkeypatch, tmp_path):
    options = parse_options(monkeypatch, '--run_dir', str(tmp_path / 'experiment'))
    definitions = training_namespace(options)
    monkeypatch.chdir(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError('Error 804: forward compatibility was attempted on non supported HW')
    monkeypatch.setattr(torch, 'empty', fail)
    with pytest.raises(SystemExit, match='Error 804'):
        exec(trainer_startup_code(), definitions)
    assert list(tmp_path.iterdir()) == []


def test_run_directory_metadata_collision_and_resume_validation(monkeypatch, tmp_path):
    checkpoint = tmp_path / 'drg.tar'
    checkpoint.write_bytes(b'fixed DRG identity')
    options = parse_options(monkeypatch, '--manualSeed', '9182', '--gamma_rel', '1',
                            '--rel_objective', 'sdga', '--rel_time_mode', 'fixed', '--rel_time_strength', '0',
                            '--run_dir', str(tmp_path / 'experiment'), '--netR_model_path', str(checkpoint))
    options.cuda = False
    monkeypatch.chdir(tmp_path)
    definitions = training_namespace(options)
    definitions.update(cudnn=torch.backends.cudnn, hashlib=hashlib)
    startup = trainer_startup_code()
    exec(startup, definitions)
    config_path = Path(options.run_dir) / 'config.json'
    metadata = json.loads(config_path.read_text(encoding='utf-8'))
    assert metadata['training']['manualSeed'] == 9182
    assert metadata['training']['drg_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert definitions['model_save_name'] == str(Path(options.run_dir) / 'dfg')
    original_config = config_path.read_bytes()
    with pytest.raises(FileExistsError, match='not empty'):
        exec(startup, definitions)
    assert config_path.read_bytes() == original_config
    options.resume_training = str(Path(options.run_dir) / 'dfg_training_last.tar')
    options.nepoch += 10
    exec(startup, definitions)  # Extending the budget does not change the objective.
    torch.save({'training_config': metadata['training']}, options.resume_training)
    options.manualSeed = None
    exec(startup, definitions)
    assert options.manualSeed == 9182
    options.rel_generator_class_weight = 0
    with pytest.raises(ValueError, match='configuration differs'):
        exec(startup, definitions)
    options.rel_generator_class_weight = 1
    checkpoint.write_bytes(b'a different DRG at the same path')
    with pytest.raises(ValueError, match='configuration differs'):
        exec(startup, definitions)
