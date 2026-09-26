"""Numerical and integration contracts for SDGA state reweighting."""

import pytest
import torch
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

from relation.timestep_schedule import state_reweighting_schedule
from relation.losses import reduce_valid_blocks, sdga_pair_losses
from relation.alignment import StateAwareRelationAlignment
from test_relation_training import parse_options


def test_full_schedule_weights_and_granularity_swap():
    signal = torch.tensor([1.0, 0.5, 0.01, 0.0], requires_grad=True)
    weights = state_reweighting_schedule(4, signal, 'granularity', 0.25, 0.5, 0.5)
    swapped = state_reweighting_schedule(4, signal, 'granularity', 0.5, 0.25, 0.5)
    for name, power in [('class', 0.25), ('instance', 0.5)]:
        raw = 0.5 + 0.5 * signal.detach().double().pow(power)
        torch.testing.assert_close(weights[name + '_raw_weight'], raw.float())
        torch.testing.assert_close(weights[name + '_weight'], (raw / raw.mean()).float())
        assert weights[name + '_weight'].mean().item() == pytest.approx(1)
        assert not weights[name + '_weight'].requires_grad
        assert torch.all(weights[name + '_weight'][:-1] >= weights[name + '_weight'][1:])
    assert torch.equal(weights['class_weight'], swapped['instance_weight'])
    assert torch.equal(weights['instance_weight'], swapped['class_weight'])


@pytest.mark.parametrize('mode,powers,floor', [
    ('fixed', (0.25, 0.5), 0.5), ('granularity', (0, 0), 0.5),
    ('granularity', (0.25, 0.5), 1), ('shared', (0, 0), 0),
])
def test_fixed_anchors(mode, powers, floor):
    weights = state_reweighting_schedule(3, torch.tensor([1., 0.1, 0.]), mode, *powers, floor)
    for name in ('class', 'instance'):
        assert torch.equal(weights[name + '_weight'], torch.ones(3))


def test_missing_states_do_not_cancel_weight_and_empty_gradient_is_zero():
    weights = state_reweighting_schedule(3, torch.tensor([1., .1, .01]), 'shared')['class_weight']
    losses = torch.tensor([2., 5., 3.], requires_grad=True)
    valid = torch.tensor([False, True, False])
    result = reduce_valid_blocks(losses, valid, weights)
    assert result.item() == pytest.approx(5 * weights[1].item())
    assert result.item() != pytest.approx(5)
    result.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([0., weights[1], 0.]))
    losses.grad = None
    reduce_valid_blocks(losses, torch.zeros(3, dtype=torch.bool), weights).backward()
    assert torch.equal(losses.grad, torch.zeros(3))


def test_tiny_signal_and_single_state_are_finite():
    result = state_reweighting_schedule(3, torch.tensor([1e-30, 1e-35, 0.]),
                                        'granularity', 0.5, 2., 0)
    assert all(torch.isfinite(value).all() for value in result.values())
    one = state_reweighting_schedule(1, torch.tensor([.001]), 'shared')
    assert one['class_weight'].item() == one['instance_weight'].item() == 1


@pytest.mark.parametrize('signal,mode,power,floor', [
    (None, 'shared', .5, .5), (torch.tensor([float('nan')]), 'shared', .5, .5),
    (torch.tensor([1.1]), 'shared', .5, .5), (torch.tensor([-.1]), 'shared', .5, .5),
    (torch.tensor([.5]), 'bad', .5, .5), (torch.tensor([.5]), 'shared', -1, .5),
    (torch.tensor([.5]), 'shared', float('inf'), .5),
    (torch.tensor([.5]), 'shared', .5, float('nan')),
    (torch.tensor([0.]), 'shared', .5, 0),
])
def test_invalid_schedules_fail(signal, mode, power, floor):
    with pytest.raises(ValueError):
        state_reweighting_schedule(1, signal, mode, power, power, floor)


def make_model(mode='fixed', **kwargs):
    return StateAwareRelationAlignment(n_timesteps=3, visual_dim=4, contrastive_dim=4, projection_dim=5,
                         objective='sdga', time_pair_weight=1, time_mode='fixed',
                         time_strength=0, topology_norm='timestep',
                         signal_retention=torch.tensor([.9, .1, .001]), gsr_mode=mode, **kwargs)


@pytest.mark.parametrize('mode,powers,floor', [
    ('fixed', (.5, .5), .5), ('shared', (.5, .5), .5),
    ('granularity', (.25, .5), .5), ('granularity', (0, 0), .5),
    ('granularity', (.25, .5), 1),
])
def test_sdga_integration_gradients_and_calibration(mode, powers, floor):
    torch.manual_seed(81)
    model = make_model(mode, gsr_class_power=powers[0], gsr_instance_power=powers[1],
                       gsr_floor=floor, generator_class_weight=.5, generator_instance_weight=1.5)
    anchor = make_model(generator_class_weight=.5, generator_instance_weight=1.5)
    anchor.load_state_dict(model.state_dict(), strict=True)
    visual = torch.randn(12, 4, requires_grad=True)
    con = torch.randn(12, 4, requires_grad=True)
    semantic = torch.randn(12, 3, requires_grad=True)
    labels = torch.arange(6).repeat_interleave(2)
    states = torch.tensor([0] * 6 + [2] * 6)  # State 1 is absent.
    assert torch.equal(model.calibration_losses(visual, semantic, con)['total'],
                       anchor.calibration_losses(visual, semantic, con)['total'])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = sdga_pair_losses(model.visual_projector(visual), semantic,
                              model.contrastive_projector(con.detach()), labels, states, 3)
    output = model(visual, semantic, con, labels, states)
    profile = model.gsr_profile()
    expected = sum(coef * (blocks[name + '_loss_per_block'][[0, 2]]
                           * profile[name + '_weight'][[0, 2]]).mean()
                   for name, coef in [('class', .5), ('instance', 1.5)])
    torch.testing.assert_close(output['total'], expected)
    contributions = sum(output['blocks'][name + '_contribution_per_block'].sum()
                        for name in ('class', 'instance'))
    torch.testing.assert_close(output['total'], contributions)
    if mode == 'fixed' or powers == (0, 0) or floor == 1:
        old_loss = .5 * blocks['class'] + 1.5 * blocks['instance']
        assert torch.equal(output['total'], old_loss)
        old_grad = torch.autograd.grad(old_loss, visual, retain_graph=True)[0]
        new_grad = torch.autograd.grad(output['total'], visual, retain_graph=True)[0]
        assert torch.equal(old_grad, new_grad)
    output['total'].backward()
    assert torch.isfinite(visual.grad).all() and visual.grad.norm() > 0
    assert con.grad is None and semantic.grad is None
    assert all(p.grad is None for p in model.parameters())
    if mode != 'fixed':
        with pytest.raises(ValueError, match='matched'):
            model(visual, semantic, con, labels, states, (states + 1) % 3)


@pytest.mark.parametrize('extra', [
    ['--rel_gsr_class_power', 'nan'], ['--rel_gsr_instance_power', '-1'],
    ['--rel_gsr_floor', 'inf'], ['--rel_gsr_floor', '1.1'],
    ['--rel_gsr_mode', 'shared', '--rel_gsr_class_power', '.25'],
    ['--rel_gsr_mode', 'granularity', '--rel_pair_grouping', 'mixed',
     '--g_batch_mode', 'pk', '--g_timestep_policy', 'class_group'],
    ['--rel_gsr_mode', 'shared', '--gamma_rel', '0'],
    ['--rel_gsr_mode', 'shared', '--run_dir', ''],
])
def test_invalid_gsr_options(monkeypatch, extra):
    with pytest.raises(SystemExit):
        parse_options(monkeypatch, '--rel_objective', 'sdga', '--rel_time_mode', 'fixed',
                      '--rel_time_strength', '0', '--gamma_rel', '1', '--run_dir', 'test', *extra)


@pytest.mark.parametrize('dataset', ['awa2', 'cub', 'sun'])
def test_documented_gsr_commands_reach_trainer(monkeypatch, tmp_path, dataset):
    root = Path(__file__).resolve().parents[1]
    commands = [shlex.split(line)[2:] for line in (root / 'SERVER_RUNBOOK.md').read_text(encoding='utf-8').splitlines()
                if line.startswith(f'python scripts/run_{dataset}_zerodiff_DFG_train.py ')
                and '--rel_gsr_mode' in line]
    assert len(commands) == 4
    checkpoint = tmp_path / 'drg.tar'
    checkpoint.touch()
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kwargs: calls.append(cmd))
    for arguments, mode, powers in zip(commands, ['fixed', 'shared', 'granularity', 'granularity'],
                                       [(.5, .5), (.5, .5), (.25, .5), (.5, .25)]):
        monkeypatch.setattr(sys, 'argv', ['launcher', *arguments, '--netR_model_path', str(checkpoint)])
        runpy.run_path(str(root / 'scripts' / f'run_{dataset}_zerodiff_DFG_train.py'))
        options = parse_options(monkeypatch, *calls[-1][2:])
        assert options.rel_gsr_mode == mode
        assert (options.rel_gsr_class_power, options.rel_gsr_instance_power) == powers
        assert options.rel_gsr_floor == .5 and options.rel_pair_grouping == 'matched'
        assert options.rel_generator_class_weight == options.rel_generator_instance_weight == 1
        assert options.run_dir and options.netR_model_path == str(checkpoint)
