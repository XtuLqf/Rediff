# author: ZihanYe
# ZeroDiff (ICLR25)
from __future__ import print_function
import faulthandler
import gc
import math
import json
import hashlib
import random
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
# import functions
import datasets.image_util as util
import classifiers.classifier_images as classifier
from config_zerodiff import opt
import zerodiff_tools
from sklearn import preprocessing
import numpy as np
import os
from relation import TimeAwareVSRA, sample_relation_group_timesteps
from relation.timestep_schedule import mixed_relation_groups

faulthandler.enable(all_threads=True)

class Logger(object):
    def __init__(self, filename, append=False):
        self.filename = filename
        f = open(self.filename + '.log', "a" if append else "w")
        f.close()

    def write(self, message):
        f = open(self.filename + '.log', "a")
        f.write(message)
        f.close()

def ensure_cuda_ready():
    """Check the actual CUDA context before creating logs or loading data."""
    if not opt.cuda:
        return
    try:
        torch.empty(1, device='cuda')
    except (RuntimeError, AssertionError) as error:
        raise SystemExit(
            'CUDA 初始化失败，训练尚未开始，未创建本次输出文件。\n'
            f'{error}\n'
            '请先核对 nvidia-smi、torch.version.cuda 和 LD_LIBRARY_PATH。'
        ) from None


ensure_cuda_ready()

folder_name = "./log"
os.makedirs(folder_name, exist_ok=True)
folder_name = "./out"
os.makedirs(folder_name, exist_ok=True)
os.makedirs(f"./log/{opt.dataset}", exist_ok=True)
os.makedirs(f"./out/{opt.dataset}", exist_ok=True)

relation_run_config = {
    'gamma': opt.gamma_rel,
    'class_weight': opt.rel_class_weight,
    'instance_weight': opt.rel_instance_weight,
    'projection_dim': opt.rel_proj_dim,
    'teacher_anchor_weight': opt.rel_teacher_anchor_weight,
    'distance_ratio': opt.rel_dist_ratio,
    'angle_ratio': opt.rel_angle_ratio if opt.rel_use_angle else 0.0,
    'time_pair_weight': opt.rel_time_pair_weight,
    'time_mode': opt.rel_time_mode,
    'time_strength': opt.rel_time_strength,
    'reliability_floor': opt.rel_reliability_floor,
    'topology_norm': opt.rel_topology_norm,
    'objective': opt.rel_objective,
    'generator_class_weight': opt.rel_generator_class_weight,
    'generator_instance_weight': opt.rel_generator_instance_weight,
    'g_batch_mode': opt.g_batch_mode,
    'g_pk_classes': opt.g_pk_classes if opt.g_batch_mode == 'pk' else 0,
    'g_pk_samples': opt.g_pk_samples if opt.g_batch_mode == 'pk' else 0,
    'g_timestep_policy': opt.g_timestep_policy,
    'pair_grouping': opt.rel_pair_grouping,
}


def normalized_relation_config(config):
    """Expand old checkpoint defaults without allowing a new objective on resume."""
    config = dict(config or {})
    defaults = {
        'objective': 'legacy', 'generator_class_weight': config.get('class_weight'),
        'generator_instance_weight': config.get('instance_weight'),
        'g_batch_mode': 'random', 'g_pk_classes': 0, 'g_pk_samples': 0,
        'g_timestep_policy': 'auto', 'pair_grouping': 'matched',
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    return config


new_relation_behavior = (
    opt.rel_objective != 'legacy' or opt.g_batch_mode != 'random'
    or opt.g_timestep_policy != 'auto'
    or opt.rel_generator_class_weight != opt.rel_class_weight
    or opt.rel_generator_instance_weight != opt.rel_instance_weight
)
if new_relation_behavior and not opt.run_dir:
    raise ValueError('New relation/sampling settings require --run_dir to isolate experiments.')
if opt.run_dir:
    opt.run_dir = os.path.abspath(opt.run_dir)
    if os.path.isdir(opt.run_dir) and os.listdir(opt.run_dir) and not opt.resume_training:
        raise FileExistsError('Run directory is not empty; use a new --run_dir or --resume_training.')
    os.makedirs(opt.run_dir, exist_ok=True)
if opt.gamma_rel > 0:
    relation_run_suffix = (
        f"_tvsra-g{opt.gamma_rel:g}"
        f"-{opt.rel_time_mode}"
        f"-s{opt.rel_time_strength:g}"
        f"-rf{opt.rel_reliability_floor:g}"
        f"-n{opt.rel_topology_norm}"
        f"-c{opt.rel_class_weight:g}"
        f"-i{opt.rel_instance_weight:g}"
        f"-p{opt.rel_proj_dim}"
        f"-tp{opt.rel_time_pair_weight:g}"
    )
else:
    relation_run_suffix = ""

logger_name = "./log/%s/train_zerodiff_DFG_%dpercent_att:%s_b:%d_lr:%s_n_T:%d_betas:%s,%s_gamma:ADV:%.1f_VAE:%.1f_x0:%.1f_xt:%.1f_dist:%.1f_f:%.1f_num:%s" % (
    opt.dataset, opt.split_percent, opt.class_embedding, opt.batch_size, str(opt.lr), opt.n_T, str(opt.ddpmbeta1),
    str(opt.ddpmbeta2), opt.gamma_ADV, opt.gamma_VAE, opt.gamma_x0, opt.gamma_xt, opt.gamma_dist, opt.factor_dist, opt.syn_num) + relation_run_suffix
model_save_name = "./out/%s/zerodiff_DFG_%dpercent_att:%s_b:%d_lr:%s_n_T:%d_betas:%s,%s_gamma:ADV:%.1f_VAE:%.1f_x0:%.1f_xt:%.1f_dist:%.1f_f:%.1f_num:%d" % (
    opt.dataset, opt.split_percent, opt.class_embedding, opt.batch_size, str(opt.lr), opt.n_T, str(opt.ddpmbeta1),
    str(opt.ddpmbeta2), opt.gamma_ADV, opt.gamma_VAE, opt.gamma_x0, opt.gamma_xt, opt.gamma_dist, opt.factor_dist, opt.syn_num) + relation_run_suffix
if opt.run_dir:
    logger_name = os.path.join(opt.run_dir, 'train')
    model_save_name = os.path.join(opt.run_dir, 'dfg')
logger = Logger(logger_name, append=bool(opt.resume_training))
if opt.gamma_rel > 0:
    relation_config_record = "TimeAwareVSRA configuration: " + str(relation_run_config)
    print(relation_config_record)
    logger.write(relation_config_record + '\n')


if opt.manualSeed is None and opt.resume_training:
    # Direct invocations may originally have let the trainer choose the seed.
    # Recover that choice before comparing the complete training configuration.
    seed_checkpoint = torch.load(opt.resume_training, map_location='cpu', weights_only=False)
    opt.manualSeed = seed_checkpoint.get('training_config', {}).get('manualSeed')
    del seed_checkpoint
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
# Legacy keeps its original RNG behavior; isolated runs also seed NumPy.
if opt.run_dir:
    np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = False
cudnn.deterministic = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# load data
training_run_config = {
    key: value for key, value in vars(opt).items()
    if key not in {'nepoch', 'resume_training', 'run_dir', 'training_checkpoint_interval', 'workers'}
}
for path_key in ('netR_model_path', 'dataroot'):
    if training_run_config[path_key] is not None:
        training_run_config[path_key] = os.path.abspath(training_run_config[path_key])
if opt.run_dir and opt.netR_model_path:
    drg_digest = hashlib.sha256()
    with open(opt.netR_model_path, 'rb') as drg_file:
        for chunk in iter(lambda: drg_file.read(1024 * 1024), b''):
            drg_digest.update(chunk)
    training_run_config['drg_sha256'] = drg_digest.hexdigest()
if opt.run_dir:
    config_path = os.path.join(opt.run_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, encoding='utf-8') as config_file:
            previous_config = json.load(config_file)
        if previous_config['training'] != training_run_config:
            raise ValueError('Run directory configuration differs from the requested training configuration.')
    else:
        with open(config_path, 'w', encoding='utf-8') as config_file:
            json.dump({'training': training_run_config, 'relation': relation_run_config,
                       'requested_epochs': opt.nepoch}, config_file, ensure_ascii=False, indent=2)
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

###########
# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_con = torch.FloatTensor(opt.batch_size, 2048)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)  # attSize class-embedding size
input_label = torch.LongTensor(opt.batch_size)  # attSize class-embedding size
noise = torch.FloatTensor(opt.batch_size, opt.noiseSize)
input_test_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_test_con = torch.FloatTensor(opt.batch_size, opt.resSize)
input_test_att = torch.FloatTensor(opt.batch_size, opt.attSize)
##########
# Cuda
if opt.cuda:
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    input_label = input_label.cuda()
    input_con = input_con.cuda()
    input_test_res, input_test_con, input_test_att = input_test_res.cuda(), input_test_con.cuda(), input_test_att.cuda()


def loss_fn(recon_x, x, mean, log_var):
    Recon = torch.nn.functional.binary_cross_entropy(
        recon_x + 1e-12, x.detach(), reduction="sum"
    )
    Recon = Recon.sum() / x.size(0)
    # Recon = torch.nn.functional.mse_loss(recon_x, x.detach(), size_average=False)
    # Recon = Recon.sum() / x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return (Recon + KLD)


def sample(batch_size):
    batch_feature, batch_con, batch_att, batch_label = data.next_seen_batch(batch_size)
    input_res.copy_(batch_feature)
    input_con.copy_(batch_con)
    input_att.copy_(batch_att)
    input_label.copy_(batch_label)
    return input_res, input_con, input_att, input_label


def sample_con(batch_size):
    idx = torch.randperm(data.ntrain)[0:batch_size]
    batch_feature = data.train_feature[idx]
    batch_label = data.train_label[idx]
    batch_att = data.attribute[batch_label]

    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(batch_label)

    return input_res, input_att, input_label

def sampleTestSeen():
    batch_feature, batch_con, batch_att, _ = data.next_test_seen_batch(opt.batch_size)
    input_test_res.copy_(batch_feature)
    input_test_con.copy_(batch_con)
    input_test_att.copy_(batch_att)

    return input_test_res, input_test_con, input_test_att

def WeightedL14att(pred, gt):
    wt = (pred - gt).pow(2)
    scale = wt.sum(1).sqrt().clamp_min(1e-12)
    wt /= scale.unsqueeze(1).expand(wt.size(0), wt.size(1))
    loss = wt * (pred - gt).abs()
    return loss.sum() / loss.size(0)

def generate_syn_feature(zerodiff, classes, attribute, num, progressive=False):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_con = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    if opt.cuda:
        syn_att = syn_att.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        with torch.no_grad():
            fake, fake_con = zerodiff.sample_from_model(syn_att, progressive=progressive)

        output = fake
        output_con = fake_con
        syn_feature.narrow(0, i * num, num).copy_(output.detach().cpu())
        syn_con.narrow(0, i * num, num).copy_(output_con.detach().cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_con, syn_label


def save_zerodiff(zerodiff, save_name, post):
    checkpoint = {'state_dict_E': zerodiff.netE.state_dict(),
                  'state_dict_G': zerodiff.netG.state_dict(),
                  'state_dict_Dec': zerodiff.netDec.state_dict(),
                  'state_dict_D_x0': zerodiff.netD_x0.state_dict(),
                  'state_dict_D_xt': zerodiff.netD_xt.state_dict(),
                  'state_dict_D_xc': zerodiff.netD_xc.state_dict(),
                  }
    if zerodiff.relationship_enabled:
        checkpoint['state_dict_VSRA'] = zerodiff.time_aware_vsra.state_dict()
        checkpoint['method_metadata'] = {
            'name': 'ds_reg_sdga' if opt.rel_objective == 'sdga' else 'time_aware_vsra',
            'config': relation_run_config,
        }
    separator = '_' if opt.run_dir else ''
    torch.save(checkpoint, save_name + separator + post + '.tar')


BEST_STATE_NAMES = (
    'best_gzsl_acc_V', 'best_acc_seen_V', 'best_acc_unseen_V', 'best_zsl_acc_V',
    'best_gzsl_acc_VS', 'best_acc_seen_VS', 'best_acc_unseen_VS', 'best_zsl_acc_VS',
    'best_gzsl_acc_C', 'best_acc_seen_C', 'best_acc_unseen_C', 'best_zsl_acc_C',
    'best_gzsl_acc_VC', 'best_acc_seen_VC', 'best_acc_unseen_VC', 'best_zsl_acc_VC',
    'best_gzsl_acc_VCS', 'best_acc_seen_VCS', 'best_acc_unseen_VCS', 'best_zsl_acc_VCS',
    'best_seen_acc_V',
    'best_acc_seen_list_V', 'best_acc_unseen_list_V', 'best_acc_zsl_list_V',
    'best_acc_seen_list_C', 'best_acc_unseen_list_C', 'best_acc_zsl_list_C',
    'best_acc_seen_list_VC', 'best_acc_unseen_list_VC', 'best_acc_zsl_list_VC',
    'best_acc_seen_list_VS', 'best_acc_unseen_list_VS', 'best_acc_zsl_list_VS',
    'best_acc_seen_list_VCS', 'best_acc_unseen_list_VCS', 'best_acc_zsl_list_VCS',
)


def save_training_state(zerodiff, save_name, epoch):
    """Atomically save enough state to resume after a native process failure."""
    checkpoint = {
        'next_epoch': epoch + 1,
        'relation_config': relation_run_config,
        'training_config': training_run_config,
        'generator_rng_states': {key: rng.get_state() for key, rng in zerodiff.generator_rngs.items()},
        'state_dict_E': zerodiff.netE.state_dict(),
        'state_dict_G': zerodiff.netG.state_dict(),
        'state_dict_Dec': zerodiff.netDec.state_dict(),
        'state_dict_D_x0': zerodiff.netD_x0.state_dict(),
        'state_dict_D_xt': zerodiff.netD_xt.state_dict(),
        'state_dict_D_xc': zerodiff.netD_xc.state_dict(),
        'optimizer_E': zerodiff.optimizerE.state_dict(),
        'optimizer_G': zerodiff.optimizerG.state_dict(),
        'optimizer_Dec': zerodiff.optimizerDec.state_dict(),
        'optimizer_D_x0': zerodiff.optimizerD_x0.state_dict(),
        'optimizer_D_xt': zerodiff.optimizerD_xt.state_dict(),
        'optimizer_D_xc': zerodiff.optimizerD_xc.state_dict(),
        'lambda1': zerodiff.lambda1,
        'best_state': {name: globals()[name] for name in BEST_STATE_NAMES},
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': (
            torch.cuda.get_rng_state(zerodiff.device) if opt.cuda else None
        ),
    }
    if zerodiff.relationship_enabled:
        checkpoint['state_dict_VSRA'] = zerodiff.time_aware_vsra.state_dict()
        checkpoint['optimizer_VSRA'] = zerodiff.optimizerVSRA.state_dict()
    checkpoint_path = save_name + '_training_last.tar'
    temporary_path = checkpoint_path + '.tmp'
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    return checkpoint_path


def load_training_state(zerodiff, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=zerodiff.device,
        weights_only=False,
    )
    if normalized_relation_config(checkpoint.get('relation_config')) != relation_run_config:
        raise ValueError(
            "Resume checkpoint relation configuration does not match this run."
        )
    saved_config = checkpoint.get('training_config')
    if saved_config is not None and saved_config != training_run_config:
        differences = sorted(key for key in set(saved_config) | set(training_run_config)
                             if saved_config.get(key) != training_run_config.get(key))
        raise ValueError('Resume training configuration differs: ' + ', '.join(differences))
    rng_states = checkpoint.get('generator_rng_states')
    if rng_states is None and new_relation_behavior:
        raise ValueError('New sampling/objective runs require generator RNG states in the checkpoint.')
    if rng_states is not None:
        if set(rng_states) != set(zerodiff.generator_rngs):
            raise ValueError('Checkpoint generator RNG streams do not match.')
        for key, rng in zerodiff.generator_rngs.items():
            rng.set_state(rng_states[key].cpu())
    zerodiff.netE.load_state_dict(checkpoint['state_dict_E'])
    zerodiff.netG.load_state_dict(checkpoint['state_dict_G'])
    zerodiff.netDec.load_state_dict(checkpoint['state_dict_Dec'])
    zerodiff.netD_x0.load_state_dict(checkpoint['state_dict_D_x0'])
    zerodiff.netD_xt.load_state_dict(checkpoint['state_dict_D_xt'])
    zerodiff.netD_xc.load_state_dict(checkpoint['state_dict_D_xc'])
    zerodiff.optimizerE.load_state_dict(checkpoint['optimizer_E'])
    zerodiff.optimizerG.load_state_dict(checkpoint['optimizer_G'])
    zerodiff.optimizerDec.load_state_dict(checkpoint['optimizer_Dec'])
    zerodiff.optimizerD_x0.load_state_dict(checkpoint['optimizer_D_x0'])
    zerodiff.optimizerD_xt.load_state_dict(checkpoint['optimizer_D_xt'])
    zerodiff.optimizerD_xc.load_state_dict(checkpoint['optimizer_D_xc'])
    zerodiff.lambda1 = checkpoint['lambda1']
    if zerodiff.relationship_enabled:
        zerodiff.time_aware_vsra.load_state_dict(checkpoint['state_dict_VSRA'])
        zerodiff.optimizerVSRA.load_state_dict(checkpoint['optimizer_VSRA'])
    globals().update(checkpoint.get('best_state', {}))
    random.setstate(checkpoint['python_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
    cuda_rng_state = checkpoint.get('cuda_rng_state')
    if opt.cuda and cuda_rng_state is not None:
        # v1.06 checkpoints stored one state per visible GPU. Loading with
        # map_location='cuda' also moves these ByteTensors to CUDA, while
        # torch.cuda.set_rng_state requires a CPU ByteTensor.
        if isinstance(cuda_rng_state, (list, tuple)):
            if not cuda_rng_state:
                raise ValueError('Checkpoint contains an empty CUDA RNG state list.')
            device_index = torch.cuda.current_device()
            state_index = device_index if device_index < len(cuda_rng_state) else 0
            cuda_rng_state = cuda_rng_state[state_index]
        torch.cuda.set_rng_state(cuda_rng_state.cpu(), device=zerodiff.device)
    return int(checkpoint['next_epoch'])


class ZERODIFF(torch.nn.Module):
    def __init__(self, data, n_T, betas, seenclasses, unseenclasses, attribute,  netR_model_path, device='cuda'):
        super(ZERODIFF, self).__init__()
        self.n_T = n_T
        self.dim_v = opt.resSize
        self.dim_s = opt.attSize
        self.dim_noise = opt.noiseSize
        self.device = device
        self.seenclasses = seenclasses
        self.unseenclasses = unseenclasses
        self.attribute = attribute
        self.attribute_seen = attribute[data.seenclasses]

        self.prior_coefficients = zerodiff_tools.ddpmgan_prior_coefficients(betas[0], betas[1], n_T, device, False)
        self.posterior_coefficients = zerodiff_tools.ddpmgan_posterior_coefficients(betas[0], betas[1], n_T, device, False)

        self.netE = zerodiff_tools.Encoder(opt).to(self.device)
        self.netG = zerodiff_tools.DFG_Generator(opt).to(self.device)
        self.netD_x0 = zerodiff_tools.DFG_Discriminator_x0(opt).to(self.device)
        self.netD_xt = zerodiff_tools.DFG_Discriminator_xt(opt).to(self.device)
        self.netD_xc = zerodiff_tools.DFG_Discriminator_xc(opt).to(self.device)
        self.netDec = zerodiff_tools.V2S_mapping(opt, opt.attSize).to(self.device)

        self.optimizerE = optim.Adam(self.netE.parameters(), lr=opt.lr)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerDec = optim.Adam(self.netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
        self.optimizerD_x0 = optim.Adam(self.netD_x0.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerD_xt = optim.Adam(self.netD_xt.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerD_xc = optim.Adam(self.netD_xc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.gamma_ADV = opt.gamma_ADV
        self.gamma_VAE = opt.gamma_VAE
        self.gamma_x0 = opt.gamma_x0
        self.gamma_xt = opt.gamma_xt
        self.gamma_recons = opt.gamma_recons
        self.lambda1 = opt.lambda1
        self.gamma_dist = opt.gamma_dist
        self.factor_dist = opt.factor_dist
        self.gamma_rel = opt.gamma_rel
        self.relationship_enabled = self.gamma_rel > 0
        self.time_aware_vsra = None
        self.optimizerVSRA = None
        if self.relationship_enabled:
            self.time_aware_vsra = TimeAwareVSRA(
                n_timesteps=self.n_T,
                visual_dim=self.dim_v,
                contrastive_dim=2048,
                projection_dim=opt.rel_proj_dim,
                class_weight=opt.rel_class_weight,
                instance_weight=opt.rel_instance_weight,
                teacher_anchor_weight=opt.rel_teacher_anchor_weight,
                distance_ratio=opt.rel_dist_ratio,
                angle_ratio=opt.rel_angle_ratio if opt.rel_use_angle else 0.0,
                angle_max_samples=opt.rel_angle_max_samples,
                time_pair_weight=opt.rel_time_pair_weight,
                time_mode=opt.rel_time_mode,
                time_strength=opt.rel_time_strength,
                signal_retention=(
                    self.prior_coefficients.sqrt_alphas_bar[1:self.n_T + 1]
                    .square()
                    .detach()
                ),
                reliability_floor=opt.rel_reliability_floor,
                topology_norm=opt.rel_topology_norm,
                objective=opt.rel_objective,
                generator_class_weight=opt.rel_generator_class_weight,
                generator_instance_weight=opt.rel_generator_instance_weight,
            ).to(self.device)
            self.optimizerVSRA = optim.Adam(
                self.time_aware_vsra.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )

        self.loss_mse = torch.nn.MSELoss(reduction="none")

        self.batch_size = opt.batch_size
        self.data = data
        # Independent streams keep mixed grouping from shifting future G/D noise,
        # minibatches or time assignments. They are persisted in full checkpoints.
        self.generator_rngs = {
            name: torch.Generator().manual_seed(opt.manualSeed + offset)
            for name, offset in [('batch', 101), ('timestep', 211), ('grouping', 307)]
        }
        if opt.g_batch_mode == 'pk':
            pk_info = self.data.prepare_pk_sampler(opt.g_pk_classes, opt.g_pk_samples)
            pk_record = 'Generator PK sampler: ' + str(pk_info)
            print(pk_record)
            logger.write(pk_record + '\n')

        self.netR = zerodiff_tools.DRG_Generator(opt).to(self.device)
        netR_state_dict = torch.load(
            netR_model_path,
            map_location=self.device,
            weights_only=True,
        )
        netR_weights = netR_state_dict.get('state_dict_G_con') or netR_state_dict.get('state_dict_R')
        if netR_weights is None:
            raise KeyError("netR checkpoint must contain 'state_dict_G_con' or 'state_dict_R'.")
        self.netR.load_state_dict(netR_weights)
        self.netR.eval()

        self.interval_recorder_sum = {}
        self.init_recorder()

    def init_recorder(self):
        self.relation_block_sums = {}
        self.relation_block_updates = 0
        self.interval_recorder_sum['criticD_train_real_x0'] = 0.0
        self.interval_recorder_sum['criticD_train_real_xt'] = 0.0
        self.interval_recorder_sum['criticD_train_real_xc'] = 0.0
        self.interval_recorder_sum['criticD_train_fake_x0'] = 0.0
        self.interval_recorder_sum['criticD_train_fake_xt'] = 0.0
        self.interval_recorder_sum['criticD_train_fake_xc'] = 0.0
        self.interval_recorder_sum['criticD_test_real_x0'] = 0.0
        self.interval_recorder_sum['criticD_test_real_xt'] = 0.0
        self.interval_recorder_sum['criticD_test_real_xc'] = 0.0
        self.interval_recorder_sum['rel_semantic_loss'] = 0.0
        self.interval_recorder_sum['rel_contrastive_loss'] = 0.0
        self.interval_recorder_sum['rel_pair_class_loss'] = 0.0
        self.interval_recorder_sum['rel_pair_instance_loss'] = 0.0
        self.interval_recorder_sum['rel_class_pairs'] = 0.0
        self.interval_recorder_sum['rel_instance_pairs'] = 0.0
        self.interval_recorder_sum['rel_legacy_loss'] = 0.0
        self.interval_recorder_sum['rel_real_semantic_loss'] = 0.0
        self.interval_recorder_sum['rel_real_contrastive_loss'] = 0.0
        self.interval_recorder_sum['rel_teacher_anchor_loss'] = 0.0
        self.interval_recorder_sum['rel_total_loss'] = 0.0
        self.interval_recorder_sum['rel_class_t_weight'] = 0.0
        self.interval_recorder_sum['rel_instance_t_weight'] = 0.0
        self.interval_recorder_sum['rel_timestep'] = 0.0

    def forward(self):
        gp_sum = 0  # lAMBDA VARIABLE
        for iter_d in range(opt.critic_iter):
            x_0_real, con_0_real, att_0_real, label = sample(self.batch_size)
            D_cost, Wasserstein_D, gp_sum, distill_loss = self.update_D(x_0_real, con_0_real, att_0_real, gp_sum, label)

        gp_sum /= (self.gamma_ADV * self.lambda1 * opt.critic_iter)
        if gp_sum > 1.05:
            self.lambda1 *= 1.1
        elif gp_sum < 1.001:
            self.lambda1 /= 1.1
        G_cost, vae_loss_seen = self.update_G(
            x_0_real,
            con_0_real,
            att_0_real,
            label,
        )
        return D_cost, Wasserstein_D, distill_loss, G_cost, vae_loss_seen

    def update_relation_space(self, x_0_real, con_0_real, att_0_real):
        """Calibrate the VSRA projectors on real data before constraining G."""
        for parameter in self.time_aware_vsra.parameters():
            parameter.requires_grad = True
        self.optimizerVSRA.zero_grad()
        losses = self.time_aware_vsra.calibration_losses(
            x_0_real,
            att_0_real,
            con_0_real,
        )
        (self.gamma_rel * losses['total']).backward()
        self.optimizerVSRA.step()
        for parameter in self.time_aware_vsra.parameters():
            parameter.requires_grad = False
        self.interval_recorder_sum['rel_real_semantic_loss'] += losses['semantic'].detach().item()
        self.interval_recorder_sum['rel_real_contrastive_loss'] += losses['contrastive'].detach().item()
        self.interval_recorder_sum['rel_teacher_anchor_loss'] += losses['teacher_anchor'].detach().item()

    def update_D(self, x_0_real, con_0_real, att_0_real, gp_sum, label):
        for p in self.netE.parameters():
            p.requires_grad = False
        for p in self.netG.parameters():
            p.requires_grad = False
        for p in self.netD_x0.parameters():
            p.requires_grad = True
        for p in self.netD_xt.parameters():
            p.requires_grad = True
        for p in self.netD_xc.parameters():
            p.requires_grad = True
        for p in self.netDec.parameters():
            p.requires_grad = True

        z, means, log_var = self.netE(x_0_real, att_0_real)

        _ts_feat = torch.randint(0, self.n_T, (self.batch_size,), dtype=torch.int64).to(self.device)
        x_t_real, x_tp1_real, ratio_x0 = self.q_sample_pairs(x_0_real, _ts_feat)

        x_0_fake = self.netG(z, att_0_real, con_0_real, x_tp1_real.detach(), _ts_feat)
        x_t_fake = self.sample_posterior(x_0_fake, x_tp1_real, _ts_feat)

        self.netD_x0.zero_grad()
        self.netD_xt.zero_grad()
        self.netD_xc.zero_grad()
        self.netDec.zero_grad()

        att_0_recons = self.netDec(x_0_real)
        R_cost = self.gamma_recons * WeightedL14att(att_0_recons, att_0_real)
        R_cost.backward()

        criticD_real_x0 = -self.netD_x0(x_0_real, att_0_real).mean() if self.gamma_x0 > 0 else torch.tensor(0.0).to(self.device)
        criticG_real_xt = -self.netD_xt(x_t_real, x_tp1_real, att_0_real, con_0_real, _ts_feat).mean() if self.gamma_xt > 0 else torch.tensor(0.0).to(self.device)
        criticD_real_xc = -self.netD_xc(x_0_real, con_0_real).mean()
        criticD_real = self.gamma_x0 * criticD_real_x0 + self.gamma_xt * criticG_real_xt + criticD_real_xc
        criticD_real = self.gamma_ADV * criticD_real
        criticD_real.backward()

        criticD_fake_x0 = self.netD_x0(x_0_fake.detach(), att_0_real).mean() if self.gamma_x0 > 0 else torch.tensor(0.0).to(self.device)
        criticG_fake_xt = self.netD_xt(x_t_fake.detach(), x_tp1_real, att_0_real, con_0_real, _ts_feat).mean() if self.gamma_xt > 0 else torch.tensor(0.0).to(self.device)
        criticD_fake_xc = self.netD_xc(x_0_fake.detach(), con_0_real).mean()
        criticD_fake = self.gamma_x0 * criticD_fake_x0 + self.gamma_xt * criticG_fake_xt + criticD_fake_xc
        criticD_fake = self.gamma_ADV * criticD_fake
        criticD_fake.backward()

        # gradient penalty
        gp_x0 = self.netD_x0.calc_gradient_penalty(x_0_real, x_0_fake.detach(), att_0_real, self.lambda1)
        gp_xt = self.netD_xt.calc_gradient_penalty(x_t_real, x_t_fake.detach(), x_tp1_real, att_0_real, con_0_real, _ts_feat, self.lambda1)
        gp_xc = self.netD_xc.calc_gradient_penalty(x_0_real, x_0_fake.detach(), con_0_real, self.lambda1)
        gp = self.gamma_ADV * (self.gamma_x0 * gp_x0 + self.gamma_xt * gp_xt + gp_xc)
        gp.backward()
        gp_sum += gp.item()
        Wasserstein_D = criticD_real - criticD_fake

        # distill
        factor = ratio_x0**self.factor_dist
        # print(factor)
        criticD_real_x0 = self.netD_x0(x_0_real, att_0_real) if self.gamma_x0 > 0 else torch.tensor(0.0).to(self.device)
        criticD_fake_x0 = self.netD_x0(x_0_fake.detach(), att_0_real) if self.gamma_x0 > 0 else torch.tensor(0.0).to(self.device)

        criticD_real_xt = self.netD_xt(x_t_real, x_tp1_real, att_0_real, con_0_real, _ts_feat) if self.gamma_xt > 0 else torch.tensor(0.0).to(self.device)
        criticD_fake_xt = self.netD_xt(x_t_fake.detach(), x_tp1_real, att_0_real, con_0_real, _ts_feat) if self.gamma_xt > 0 else torch.tensor(0.0).to(self.device)

        criticD_real_xc = self.netD_xc(x_0_real, con_0_real)
        criticD_fake_xc = self.netD_xc(x_0_fake.detach(), con_0_real)

        Wasserstein_D_x0 = criticD_real_x0 - criticD_fake_x0
        Wasserstein_D_xt = criticD_real_xt - criticD_fake_xt
        Wasserstein_D_xc = criticD_real_xc - criticD_fake_xc

        distill_loss = (factor * self.loss_mse(Wasserstein_D_x0, Wasserstein_D_xt.detach())).mean() + (factor * self.loss_mse(Wasserstein_D_xt, Wasserstein_D_x0.detach())).mean() + \
                       (factor * self.loss_mse(Wasserstein_D_xc, Wasserstein_D_xt.detach())).mean() + (factor * self.loss_mse(Wasserstein_D_xt, Wasserstein_D_xc.detach())).mean() + \
                       self.loss_mse(Wasserstein_D_x0, Wasserstein_D_xc.detach()).mean() + self.loss_mse(Wasserstein_D_xc, Wasserstein_D_x0.detach()).mean()

        distill_loss = self.gamma_dist * distill_loss
        distill_loss.backward()

        D_cost = criticD_fake - criticD_real + gp  # add Y here and #add vae reconstruction loss
        self.optimizerDec.step()
        self.optimizerD_x0.step()
        self.optimizerD_xt.step()
        self.optimizerD_xc.step()

        with torch.no_grad():
            test_seen_x_0_real, test_seen_con_0_real, test_seen_att_0_real = sampleTestSeen()
            test_seen_x_t_real, test_seen_x_tp1_real, ratio_x0 = self.q_sample_pairs(test_seen_x_0_real, _ts_feat)
            criticD_test_real_x0 = self.netD_x0(test_seen_x_0_real, test_seen_att_0_real)
            criticD_test_real_xt = self.netD_xt(test_seen_x_t_real, test_seen_x_tp1_real, test_seen_att_0_real, test_seen_con_0_real, _ts_feat)
            criticD_test_real_xc = self.netD_xc(test_seen_x_0_real, test_seen_con_0_real)

        self.interval_recorder_sum['criticD_train_real_x0'] += criticD_real_x0.detach().mean().item()
        self.interval_recorder_sum['criticD_train_real_xt'] += criticD_real_xt.detach().mean().item()
        self.interval_recorder_sum['criticD_train_real_xc'] += criticD_real_xc.detach().mean().item()
        self.interval_recorder_sum['criticD_train_fake_x0'] += criticD_fake_x0.detach().mean().item()
        self.interval_recorder_sum['criticD_train_fake_xt'] += criticD_fake_xt.detach().mean().item()
        self.interval_recorder_sum['criticD_train_fake_xc'] += criticD_fake_xc.detach().mean().item()
        self.interval_recorder_sum['criticD_test_real_x0'] += criticD_test_real_x0.detach().mean().item()
        self.interval_recorder_sum['criticD_test_real_xt'] += criticD_test_real_xt.detach().mean().item()
        self.interval_recorder_sum['criticD_test_real_xc'] += criticD_test_real_xc.detach().mean().item()

        return D_cost, Wasserstein_D, gp_sum, distill_loss

    def update_G(self, x_0_real, con_0_real, att_0_real, label):
        if self.relationship_enabled:
            self.update_relation_space(
                x_0_real,
                con_0_real,
                att_0_real,
            )
        # Calibrate on the original D batch before selecting the complete G batch.
        # Every E/G, VAE, adversarial, reconstruction and relation term uses it.
        if opt.g_batch_mode == 'pk':
            x_0_real, con_0_real, att_0_real, label = (
                tensor.to(self.device)
                for tensor in self.data.next_seen_pk_batch(self.generator_rngs['batch'])
            )
        for p in self.netE.parameters():
            p.requires_grad = True
        for p in self.netG.parameters():
            p.requires_grad = True
        for p in self.netD_x0.parameters():  # freeze discrimator
            p.requires_grad = False
        for p in self.netD_xt.parameters():
            p.requires_grad = False
        for p in self.netD_xc.parameters():
            p.requires_grad = False
        if opt.gamma_recons > 0 and opt.freeze_dec:
            for p in self.netDec.parameters():  # freeze decoder
                p.requires_grad = False

        self.netE.zero_grad()
        self.netG.zero_grad()

        z, means, log_var = self.netE(x_0_real, att_0_real)

        group_times = opt.g_timestep_policy == 'class_group' or (
            opt.g_timestep_policy == 'auto' and self.relationship_enabled
            and self.time_aware_vsra.time_pair_weight > 0
        )
        # Leave the default legacy random draws exactly where they were.
        private_rng = self.generator_rngs['timestep'] if new_relation_behavior else None
        if group_times:
            _ts_feat = sample_relation_group_timesteps(label, self.n_T, private_rng)
        else:
            _ts_feat = torch.randint(
                0,
                self.n_T,
                (x_0_real.shape[0],),
                dtype=torch.int64,
                device='cpu' if private_rng is not None else self.device,
                generator=private_rng,
            ).to(self.device)
        relation_group_ids = _ts_feat
        if opt.rel_pair_grouping == 'mixed':
            relation_group_ids = mixed_relation_groups(
                label, _ts_feat, self.n_T, self.generator_rngs['grouping'],
            )
        x_t_real, x_tp1_real, _ = self.q_sample_pairs(x_0_real, _ts_feat)
        x_0_fake = self.netG(z, att_0_real, con_0_real, x_tp1_real.detach(), _ts_feat)
        x_t_fake = self.sample_posterior(x_0_fake, x_tp1_real, _ts_feat)

        errG = 0.0
        vae_loss_seen = loss_fn(x_0_fake, x_0_real, means, log_var) if self.gamma_VAE > 0 else torch.tensor(0.0).to(self.device)
        errG += self.gamma_VAE * vae_loss_seen

        criticG_fake_x0 = -self.netD_x0(x_0_fake, att_0_real).mean() if self.gamma_x0 > 0 else torch.tensor(0.0).to( self.device)
        criticG_fake_xt = -self.netD_xt(x_t_fake, x_tp1_real, att_0_real, con_0_real, _ts_feat).mean() if self.gamma_xt > 0 else torch.tensor(0.0).to(self.device)
        criticG_fake_xc = -self.netD_xc(x_0_fake, con_0_real).mean()
        criticG_fake = self.gamma_x0 * criticG_fake_x0 + self.gamma_xt * criticG_fake_xt + criticG_fake_xc
        G_cost = criticG_fake

        errG += self.gamma_ADV * G_cost

        self.netDec.zero_grad()
        att_0_recons = self.netDec(x_0_fake)
        R_cost = WeightedL14att(att_0_recons, att_0_real)
        errG += self.gamma_recons * R_cost

        if self.relationship_enabled:
            relation_losses = self.time_aware_vsra(
                x_0_fake,
                att_0_real,
                con_0_real,
                label,
                _ts_feat,
                relation_group_ids=relation_group_ids,
            )
            if 'blocks' in relation_losses:
                self.record_relation_blocks(relation_losses['blocks'], relation_group_ids, _ts_feat)
            errG += self.gamma_rel * relation_losses['total']
            self.interval_recorder_sum['rel_semantic_loss'] += relation_losses['semantic'].detach().item()
            self.interval_recorder_sum['rel_contrastive_loss'] += relation_losses['contrastive'].detach().item()
            self.interval_recorder_sum['rel_pair_class_loss'] += relation_losses['pair_class'].detach().item()
            self.interval_recorder_sum['rel_pair_instance_loss'] += relation_losses['pair_instance'].detach().item()
            self.interval_recorder_sum['rel_class_pairs'] += relation_losses['class_pairs'].detach().item()
            self.interval_recorder_sum['rel_instance_pairs'] += relation_losses['instance_pairs'].detach().item()
            self.interval_recorder_sum['rel_legacy_loss'] += relation_losses['legacy_total'].detach().item()
            self.interval_recorder_sum['rel_total_loss'] += relation_losses['total'].detach().item()
            self.interval_recorder_sum['rel_class_t_weight'] += relation_losses['class_weight'].detach().item()
            self.interval_recorder_sum['rel_instance_t_weight'] += relation_losses['instance_weight'].detach().item()
            self.interval_recorder_sum['rel_timestep'] += _ts_feat.float().mean().detach().item()

        errG.backward()
        # write a condition here
        self.optimizerE.step()
        self.optimizerG.step()
        if self.gamma_recons > 0 and not opt.freeze_dec:  # not train decoder at feedback time
            self.optimizerDec.step()
        return G_cost, vae_loss_seen

    def record_relation_blocks(self, blocks, relation_groups, generation_timesteps):
        """Accumulate detached per-update block values; no autograd graph survives."""
        metrics = {key: value.detach().float() for key, value in blocks.items() if value.ndim > 0}
        composition = torch.bincount(
            relation_groups * self.n_T + generation_timesteps,
            minlength=self.n_T * self.n_T,
        ).reshape(self.n_T, self.n_T).float()
        metrics['generation_state_counts'] = composition
        retention = self.prior_coefficients.sqrt_alphas_bar[1:self.n_T + 1].square().detach()
        metrics['signal_retention_mean'] = (composition @ retention) / composition.sum(1).clamp_min(1)
        for key, value in metrics.items():
            if key not in self.relation_block_sums:
                self.relation_block_sums[key] = torch.zeros_like(value)
            self.relation_block_sums[key] += value
        self.relation_block_updates += 1

    def relation_block_summary(self, epoch):
        if not self.relation_block_updates:
            return None
        metrics = {key: value / self.relation_block_updates for key, value in self.relation_block_sums.items()}
        if any(not torch.isfinite(value).all() for value in metrics.values()):
            raise FloatingPointError('Non-finite SDGA block metrics at epoch %d' % epoch)
        return {'epoch': epoch, 'grouping': opt.rel_pair_grouping,
                'updates': self.relation_block_updates,
                'signal_retention_by_generation_state': self.prior_coefficients.sqrt_alphas_bar[1:self.n_T + 1].square().detach().cpu().tolist(),
                **{key: value.cpu().tolist() for key, value in metrics.items()}}

    def sample_from_model(self, att, progressive=False):
        n_sample = att.shape[0]
        with torch.no_grad():
            x_tp1 = torch.randn(n_sample, self.dim_v).to(self.device)
            z = torch.randn(n_sample, self.dim_noise).to(self.device)

            z_con = torch.randn(n_sample, self.dim_noise).to(self.device)
            _ts_con = (self.n_T - 1) + torch.zeros((n_sample,), dtype=torch.int64).to(self.device) # torch.randint(0, self.n_T + 1, (n_sample,), dtype=torch.int64).to(self.device)
            r_tp1 = torch.randn(n_sample, 2048).to(self.device)
            r_0_fake = self.netR(z_con, att, r_tp1, _ts_con)
            r_0_fake = r_0_fake.detach()

            if progressive:
                for i in reversed(range(self.n_T)):
                    _ts = torch.full((n_sample,), i, dtype=torch.int64).to(x_tp1.device)
                    x_0_pred = self.netG(z, att, r_0_fake, x_tp1, _ts)
                    x_t = self.sample_posterior(x_0_pred, x_tp1, _ts.long())
                    x_tp1 = x_t.detach()
                x_0_fake = x_0_pred.detach()
            else:
                _ts = (self.n_T - 1) + torch.zeros((n_sample,), dtype=torch.int64).to(self.device)
                x_0_fake = self.netG(z, att, r_0_fake, x_tp1, _ts)
                x_0_fake = x_0_fake.detach()

        return x_0_fake, r_0_fake

    def q_sample_pairs(self, x_0, t):
        """
        Generate a pair of disturbed images for training, use prior_coefficients
        :param x_0: x_0
        :param t: time step t
        :return: x_t, x_{t+1}
        """
        t = t.long()
        noise = torch.randn_like(x_0)
        x_t, ratio_x0 = self.q_sample(x_0, t)

        ratio_xt2xtp1 = zerodiff_tools.extract(self.prior_coefficients.sqrt_alphas, t + 1, x_0.shape)
        ratio_noise = zerodiff_tools.extract(self.prior_coefficients.sigmas, t + 1, x_0.shape)

        x_t_plus_one = ratio_xt2xtp1 * x_t + ratio_noise * noise

        return x_t, x_t_plus_one, ratio_x0

    def q_sample(self, x_0, t):
        """
        use prior_coefficients
        q(x_{t}|x_0,t)
        Diffuse the data (t == 0 means diffused for t step)
        """
        t = t.long()
        noise = torch.randn_like(x_0)

        ratio_x0 = zerodiff_tools.extract(self.prior_coefficients.sqrt_alphas_bar, t, x_0.shape)
        ratio_noise = zerodiff_tools.extract(self.prior_coefficients.sigmas_bar, t, x_0.shape)

        x_t = ratio_x0 * x_0 + ratio_noise * noise

        return x_t, ratio_x0

    def sample_posterior(self, x_0, x_t, t):
        """
        use posterior_coefficients
        q(x_{t-1}|x_0,x_t,t)
        """
        t = t.long()
        posterior_mean_coef1 = zerodiff_tools.extract(self.posterior_coefficients.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2 = zerodiff_tools.extract(self.posterior_coefficients.posterior_mean_coef2, t, x_t.shape)

        mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        log_var_clipped = zerodiff_tools.extract(self.posterior_coefficients.posterior_log_variance_clipped, t, x_t.shape)

        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        sample_x_pos = mean + nonzero_mask[:, None] * torch.exp(0.5 * log_var_clipped) * noise

        return sample_x_pos


zerodiff = ZERODIFF(data, n_T=opt.n_T, betas=(opt.ddpmbeta1, opt.ddpmbeta2), seenclasses=data.seenclasses,
                  unseenclasses=data.unseenclasses, attribute=data.attribute,
                  netR_model_path=opt.netR_model_path, device='cuda')
if zerodiff.relationship_enabled:
    profile = zerodiff.time_aware_vsra.timestep_profile()
    profile_rows = []
    for index in range(opt.n_T):
        profile_rows.append(
            "t=%d signal=%.6f snr=%.6f w_class=%.4f w_instance=%.4f"
            % (
                int(profile['timestep'][index]),
                float(profile['signal_retention'][index]),
                float(profile['snr'][index]),
                float(profile['class_weight'][index]),
                float(profile['instance_weight'][index]),
            )
        )
    profile_record = "Diffusion relation profile: " + "; ".join(profile_rows)
    print(profile_record)
    logger.write(profile_record + '\n')
zerodiff.train()

best_gzsl_acc_V = 0
best_acc_seen_V = 0
best_acc_unseen_V = 0
best_zsl_acc_V = 0

best_gzsl_acc_VS = 0
best_acc_seen_VS = 0
best_acc_unseen_VS = 0
best_zsl_acc_VS = 0

best_gzsl_acc_C = 0
best_acc_seen_C = 0
best_acc_unseen_C = 0
best_zsl_acc_C = 0

best_gzsl_acc_VC = 0
best_acc_seen_VC = 0
best_acc_unseen_VC = 0
best_zsl_acc_VC = 0

best_gzsl_acc_VCS = 0
best_acc_seen_VCS = 0
best_acc_unseen_VCS = 0
best_zsl_acc_VCS = 0

best_seen_acc_V = 0

best_acc_seen_list_V, best_acc_unseen_list_V, best_acc_zsl_list_V = [], [], []
best_acc_seen_list_C, best_acc_unseen_list_C, best_acc_zsl_list_C = [], [], []
best_acc_seen_list_VC, best_acc_unseen_list_VC, best_acc_zsl_list_VC = [], [], []
best_acc_seen_list_VS, best_acc_unseen_list_VS, best_acc_zsl_list_VS = [], [], []
best_acc_seen_list_VCS, best_acc_unseen_list_VCS, best_acc_zsl_list_VCS = [], [], []


n_iter = len(range(0, data.ntrain, opt.batch_size))
start_epoch = 0
if opt.resume_training:
    start_epoch = load_training_state(zerodiff, opt.resume_training)
    resume_record = 'Resumed full training state from %s at epoch %d' % (
        opt.resume_training,
        start_epoch,
    )
    print(resume_record)
    logger.write(resume_record + '\n')

for epoch in range(start_epoch, opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):
        D_cost, Wasserstein_D, distill_loss, G_cost, vae_loss_seen = zerodiff()

    epoch_losses = {
        'Loss_D': D_cost.detach().item(),
        'Wasserstein_dist': Wasserstein_D.detach().item(),
        'distill_loss': distill_loss.detach().item(),
        'Loss_G': G_cost.detach().item(),
        'vae_loss_seen': vae_loss_seen.detach().item(),
    }
    nonfinite = [name for name, value in epoch_losses.items() if not math.isfinite(value)]
    if nonfinite:
        raise FloatingPointError(
            'Non-finite training values at epoch %d: %s'
            % (epoch, ', '.join(nonfinite))
        )

    log_record = '[%d/%d] Loss_D: %.4f, Wasserstein_dist:%.4f, distill_loss:%.4f' % (
        epoch, opt.nepoch, epoch_losses['Loss_D'], epoch_losses['Wasserstein_dist'], epoch_losses['distill_loss'])
    print(log_record)
    logger.write(log_record + '\n')

    log_record = '[%d/%d] Loss_G: %.4f, vae_loss_seen:%.4f' % (
        epoch, opt.nepoch, epoch_losses['Loss_G'], epoch_losses['vae_loss_seen'])
    print(log_record)
    logger.write(log_record + '\n')

    criticD_train_real_x0 = zerodiff.interval_recorder_sum['criticD_train_real_x0'] / n_iter
    criticD_train_real_xt = zerodiff.interval_recorder_sum['criticD_train_real_xt'] / n_iter
    criticD_train_real_xc = zerodiff.interval_recorder_sum['criticD_train_real_xc'] / n_iter

    criticD_test_real_x0 = zerodiff.interval_recorder_sum['criticD_test_real_x0'] / n_iter
    criticD_test_real_xt = zerodiff.interval_recorder_sum['criticD_test_real_xt'] / n_iter
    criticD_test_real_xc = zerodiff.interval_recorder_sum['criticD_test_real_xc'] / n_iter

    criticD_train_fake_x0 = zerodiff.interval_recorder_sum['criticD_train_fake_x0'] / n_iter
    criticD_train_fake_xt = zerodiff.interval_recorder_sum['criticD_train_fake_xt'] / n_iter
    criticD_train_fake_xc = zerodiff.interval_recorder_sum['criticD_train_fake_xc'] / n_iter
    if zerodiff.relationship_enabled:
        relation_epoch_stats = {
            key: zerodiff.interval_recorder_sum[key] / n_iter
            for key in (
                'rel_semantic_loss',
                'rel_contrastive_loss',
                'rel_pair_class_loss',
                'rel_pair_instance_loss',
                'rel_class_pairs',
                'rel_instance_pairs',
                'rel_legacy_loss',
                'rel_real_semantic_loss',
                'rel_real_contrastive_loss',
                'rel_teacher_anchor_loss',
                'rel_total_loss',
                'rel_class_t_weight',
                'rel_instance_t_weight',
                'rel_timestep',
            )
        }
    block_summary = zerodiff.relation_block_summary(epoch)
    if block_summary is not None:
        block_record = 'SDGA blocks: ' + json.dumps(block_summary, allow_nan=False)
        print(block_record)
        logger.write(block_record + '\n')
    zerodiff.init_recorder()

    log_record = '[%d/%d] D_train_real_x0: %.6f, D_train_real_xt: %.6f, D_train_real_xc: %.6f' % (epoch, opt.nepoch, criticD_train_real_x0, criticD_train_real_xt, criticD_train_real_xc)
    print(log_record)
    logger.write(log_record + '\n')

    if zerodiff.relationship_enabled:
        log_record = (
            '[%d/%d] VSRA calibration semantic/con: %.6f/%.6f, '
            'teacher anchor: %.6f'
            % (
                epoch,
                opt.nepoch,
                relation_epoch_stats['rel_real_semantic_loss'],
                relation_epoch_stats['rel_real_contrastive_loss'],
                relation_epoch_stats['rel_teacher_anchor_loss'],
            )
        )
        print(log_record)
        logger.write(log_record + '\n')

        log_record = (
            '[%d/%d] Relation static semantic/con: %.6f/%.6f, legacy: %.6f, '
            'time_pair(C/I): %.6f/%.6f, pairs(C/I): %.1f/%.1f, total: %.6f, '
            't: %.4f, w_class(t): %.4f, w_instance(t): %.4f'
            % (
                epoch,
                opt.nepoch,
                relation_epoch_stats['rel_semantic_loss'],
                relation_epoch_stats['rel_contrastive_loss'],
                relation_epoch_stats['rel_legacy_loss'],
                relation_epoch_stats['rel_pair_class_loss'],
                relation_epoch_stats['rel_pair_instance_loss'],
                relation_epoch_stats['rel_class_pairs'],
                relation_epoch_stats['rel_instance_pairs'],
                relation_epoch_stats['rel_total_loss'],
                relation_epoch_stats['rel_timestep'],
                relation_epoch_stats['rel_class_t_weight'],
                relation_epoch_stats['rel_instance_t_weight'],
            )
        )
        print(log_record)
        logger.write(log_record + '\n')

    log_record = '[%d/%d] D_test_real_x0: %.6f, D_test_real_xt: %.6f, D_test_real_xc: %.6f' % (epoch, opt.nepoch, criticD_test_real_x0, criticD_test_real_xt, criticD_test_real_xc)
    print(log_record)
    logger.write(log_record + '\n')

    log_record = '[%d/%d] D_train_fake_x0: %.6f, D_train_fake_xt: %.6f, D_train_fake_xc: %.6f' % (epoch, opt.nepoch, criticD_train_fake_x0, criticD_train_fake_xt, criticD_train_fake_xc)
    print(log_record)
    logger.write(log_record + '\n')

    if epoch % opt.eval_interval == 0 or epoch == (opt.nepoch - 1):
        zerodiff.eval()
        syn_feature, syn_con, syn_label = generate_syn_feature(zerodiff, data.unseenclasses, data.attribute, opt.syn_num)
        syn_feature_pro, syn_con_pro, syn_label_pro = generate_syn_feature(zerodiff, data.unseenclasses, data.attribute, opt.syn_num, progressive=True)
        syn_feature_seen, syn_con_seen, syn_label_seen = generate_syn_feature(zerodiff, data.seenclasses, data.attribute, opt.syn_num)

        # Train Seen classifier in V
        # Release each classifier after consuming its metrics, before the next
        # constructor allocates feature matrices and CUDA optimizer state.
        seen_cls_V = classifier.CLASSIFIER(syn_feature_seen, util.map_label(syn_label_seen, data.seenclasses), \
                                           data, data.seenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                           opt.syn_num, cls_mode="seen")
        acc = seen_cls_V.acc
        if best_seen_acc_V < acc:
            best_seen_acc_V = acc
        log_record = 'Seen (V): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')
        del seen_cls_V

        # Generalized zero-shot learning
        if opt.gzsl:
            # Concatenate real seen features with synthesized unseen features
            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_C = torch.cat((data.train_paco, syn_con), 0)
            train_Y = torch.cat((data.train_label, syn_label), 0)
            train_X_pro = torch.cat((data.train_feature, syn_feature_pro), 0)
            train_C_pro = torch.cat((data.train_paco, syn_con_pro), 0)
            train_Y_pro = torch.cat((data.train_label, syn_label_pro), 0)
            nclass = opt.nclass_all

            # Train GZSL classifier in V
            gzsl_cls_V = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5,  25, opt.syn_num, cls_mode="GZSL")
            if best_gzsl_acc_V < gzsl_cls_V.H:
                best_acc_seen_V, best_acc_unseen_V, best_gzsl_acc_V = gzsl_cls_V.acc_seen, gzsl_cls_V.acc_unseen, gzsl_cls_V.H
                best_acc_unseen_list_V, best_acc_seen_list_V = gzsl_cls_V.best_acc_U_list, gzsl_cls_V.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_V")
            log_record = 'GZSL (V): U: %.4f, S: %.4f, H: %.4f' % (
            gzsl_cls_V.acc_unseen, gzsl_cls_V.acc_seen, gzsl_cls_V.H)
            print(log_record)
            logger.write(log_record + '\n')
            del gzsl_cls_V

            gzsl_cls_V = classifier.CLASSIFIER(train_X_pro, train_Y_pro, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                                               25, opt.syn_num, cls_mode="GZSL")
            if best_gzsl_acc_V < gzsl_cls_V.H:
                best_acc_seen_V, best_acc_unseen_V, best_gzsl_acc_V = gzsl_cls_V.acc_seen, gzsl_cls_V.acc_unseen, gzsl_cls_V.H
                best_acc_unseen_list_V, best_acc_seen_list_V = gzsl_cls_V.best_acc_U_list, gzsl_cls_V.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_V")
            log_record = 'GZSL pro (V): U: %.4f, S: %.4f, H: %.4f' % (
            gzsl_cls_V.acc_unseen, gzsl_cls_V.acc_seen, gzsl_cls_V.H)
            print(log_record)
            logger.write(log_record + '\n')
            del gzsl_cls_V

            # Train GZSL classifier in VS
            gzsl_cls_VS = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                                                25, opt.syn_num, cls_mode="GZSL", netDec=zerodiff.netDec,
                                                dec_size=opt.attSize, dec_hidden_size=4096, useS=True)
            if best_gzsl_acc_VS < gzsl_cls_VS.H:
                best_acc_seen_VS, best_acc_unseen_VS, best_gzsl_acc_VS = gzsl_cls_VS.acc_seen, gzsl_cls_VS.acc_unseen, gzsl_cls_VS.H
                best_acc_unseen_list_VS, best_acc_seen_list_VS = gzsl_cls_VS.best_acc_U_list, gzsl_cls_VS.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_VS")
            log_record = 'GZSL (VS): U: %.4f, S: %.4f, H: %.4f' % (
            gzsl_cls_VS.acc_unseen, gzsl_cls_VS.acc_seen, gzsl_cls_VS.H)
            print(log_record)
            logger.write(log_record + '\n')
            del gzsl_cls_VS

            # Train GZSL classifier in VS
            gzsl_cls_VS = classifier.CLASSIFIER(train_X_pro, train_Y_pro, data, nclass, opt.cuda, opt.classifier_lr,
                                                0.5, 25, opt.syn_num, cls_mode="GZSL", netDec=zerodiff.netDec,
                                                dec_size=opt.attSize,
                                                dec_hidden_size=4096, useS=True)
            if best_gzsl_acc_VS < gzsl_cls_VS.H:
                best_acc_seen_VS, best_acc_unseen_VS, best_gzsl_acc_VS = gzsl_cls_VS.acc_seen, gzsl_cls_VS.acc_unseen, gzsl_cls_VS.H
                best_acc_unseen_list_VS, best_acc_seen_list_VS = gzsl_cls_VS.best_acc_U_list, gzsl_cls_VS.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_VS")
            log_record = 'GZSL pro (VS): U: %.4f, S: %.4f, H: %.4f' % (
            gzsl_cls_VS.acc_unseen, gzsl_cls_VS.acc_seen, gzsl_cls_VS.H)
            print(log_record)
            logger.write(log_record + '\n')
            del gzsl_cls_VS

            # Train GZSL classifier in C
            gzsl_cls_C = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                                               25, opt.syn_num, cls_mode="GZSL", con_size=2048, _train_C = train_C, useV=False, useC=True)
            if best_gzsl_acc_C < gzsl_cls_C.H:
                best_acc_seen_C, best_acc_unseen_C, best_gzsl_acc_C = gzsl_cls_C.acc_seen, gzsl_cls_C.acc_unseen, gzsl_cls_C.H
                best_acc_unseen_list_C, best_acc_seen_list_C = gzsl_cls_C.best_acc_U_list, gzsl_cls_C.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_C")
            log_record = 'GZSL (C): U: %.4f, S: %.4f, H: %.4f' % (
                gzsl_cls_C.acc_unseen, gzsl_cls_C.acc_seen, gzsl_cls_C.H)
            print(log_record)
            logger.write(log_record + '\n')
            del gzsl_cls_C

            # Train GZSL classifier in C
            gzsl_cls_C = classifier.CLASSIFIER(train_X_pro, train_Y_pro, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                                               25, opt.syn_num, cls_mode="GZSL", con_size=2048, _train_C = train_C_pro, useV=False, useC=True)
            if best_gzsl_acc_C < gzsl_cls_C.H:
                best_acc_seen_C, best_acc_unseen_C, best_gzsl_acc_C = gzsl_cls_C.acc_seen, gzsl_cls_C.acc_unseen, gzsl_cls_C.H
                best_acc_unseen_list_C, best_acc_seen_list_C = gzsl_cls_C.best_acc_U_list, gzsl_cls_C.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_C")
            log_record = 'GZSL pro (C): U: %.4f, S: %.4f, H: %.4f' % (
                gzsl_cls_C.acc_unseen, gzsl_cls_C.acc_seen, gzsl_cls_C.H)
            print(log_record)
            logger.write(log_record + '\n')
            del gzsl_cls_C

            # Train GZSL classifier in VC
            gzsl_cls_VC = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                                                25, opt.syn_num, cls_mode="GZSL", con_size=2048, _train_C = train_C, useC=True)
            if best_gzsl_acc_VC < gzsl_cls_VC.H:
                best_acc_seen_VC, best_acc_unseen_VC, best_gzsl_acc_VC = gzsl_cls_VC.acc_seen, gzsl_cls_VC.acc_unseen, gzsl_cls_VC.H
                best_acc_unseen_list_VC, best_acc_seen_list_VC = gzsl_cls_VC.best_acc_U_list, gzsl_cls_VC.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_VC")
            log_record = 'GZSL (VC): U: %.4f, S: %.4f, H: %.4f' % (
            gzsl_cls_VC.acc_unseen, gzsl_cls_VC.acc_seen, gzsl_cls_VC.H)
            print(log_record)
            logger.write(log_record + '\n')
            del gzsl_cls_VC

            # Train GZSL classifier in VC
            gzsl_cls_VC = classifier.CLASSIFIER(train_X_pro, train_Y_pro, data, nclass, opt.cuda, opt.classifier_lr,
                                                0.5, 25, opt.syn_num, cls_mode="GZSL", con_size=2048, _train_C = train_C_pro, useC=True)
            if best_gzsl_acc_VC < gzsl_cls_VC.H:
                best_acc_seen_VC, best_acc_unseen_VC, best_gzsl_acc_VC = gzsl_cls_VC.acc_seen, gzsl_cls_VC.acc_unseen, gzsl_cls_VC.H
                best_acc_unseen_list_VC, best_acc_seen_list_VC = gzsl_cls_VC.best_acc_U_list, gzsl_cls_VC.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_VC")
            log_record = 'GZSL pro (VC): U: %.4f, S: %.4f, H: %.4f' % (
            gzsl_cls_VC.acc_unseen, gzsl_cls_VC.acc_seen, gzsl_cls_VC.H)
            print(log_record)
            logger.write(log_record + '\n')
            del gzsl_cls_VC

            # Train GZSL classifier in VCS
            gzsl_cls_VCS = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                                                 25, opt.syn_num, cls_mode="GZSL", netDec=zerodiff.netDec,
                                                 dec_size=opt.attSize, dec_hidden_size=4096, useS=True, con_size=2048, _train_C = train_C, useC=True)
            if best_gzsl_acc_VCS < gzsl_cls_VCS.H:
                best_acc_seen_VCS, best_acc_unseen_VCS, best_gzsl_acc_VCS = gzsl_cls_VCS.acc_seen, gzsl_cls_VCS.acc_unseen, gzsl_cls_VCS.H
                best_acc_unseen_list_VCS, best_acc_seen_list_VCS = gzsl_cls_VCS.best_acc_U_list, gzsl_cls_VCS.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_VCS")
            log_record = 'GZSL (VCS): U: %.4f, S: %.4f, H: %.4f' % (
                gzsl_cls_VCS.acc_unseen, gzsl_cls_VCS.acc_seen, gzsl_cls_VCS.H)
            print(log_record)
            logger.write(log_record + '\n')
            del gzsl_cls_VCS

            # Train GZSL classifier in VC
            gzsl_cls_VCS = classifier.CLASSIFIER(train_X_pro, train_Y_pro, data, nclass, opt.cuda, opt.classifier_lr,
                                                 0.5,  25, opt.syn_num, cls_mode="GZSL", netDec=zerodiff.netDec,
                                                 dec_size=opt.attSize, dec_hidden_size=4096, useS=True,
                                                 con_size=2048, _train_C = train_C_pro, useC=True)  #
            if best_gzsl_acc_VCS < gzsl_cls_VCS.H:
                best_acc_seen_VCS, best_acc_unseen_VCS, best_gzsl_acc_VCS = gzsl_cls_VCS.acc_seen, gzsl_cls_VCS.acc_unseen, gzsl_cls_VCS.H
                save_zerodiff(zerodiff, model_save_name, "gzsl")
                best_acc_unseen_list_VCS, best_acc_seen_list_VCS = gzsl_cls_VCS.best_acc_U_list, gzsl_cls_VCS.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_VCS")

            log_record = 'GZSL pro (VCS): U: %.4f, S: %.4f, H: %.4f' % (
                gzsl_cls_VCS.acc_unseen, gzsl_cls_VCS.acc_seen, gzsl_cls_VCS.H)
            print(log_record)
            logger.write(log_record + '\n')
            del gzsl_cls_VCS

        # Zero-shot learning
        # Train ZSL classifier in V
        zsl_cls_V = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                                          data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                          opt.syn_num, cls_mode="ZSL")
        acc = zsl_cls_V.acc
        if best_zsl_acc_V < acc:
            best_zsl_acc_V = acc
            best_acc_zsl_list_V = zsl_cls_V.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_V")
        log_record = 'ZSL (V): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')
        del zsl_cls_V

        # Train ZSL classifier in V
        zsl_cls_V = classifier.CLASSIFIER(syn_feature_pro, util.map_label(syn_label_pro, data.unseenclasses), \
                                          data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                          opt.syn_num,  cls_mode="ZSL")

        acc = zsl_cls_V.acc
        if best_zsl_acc_V < acc:
            best_zsl_acc_V = acc
            best_acc_zsl_list_V = zsl_cls_V.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_V")
        log_record = 'ZSL pro (V): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')
        del zsl_cls_V

        # Train ZSL classifier in VS
        zsl_cls_VS = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                                           data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                           opt.syn_num, cls_mode="ZSL", netDec=zerodiff.netDec, dec_size=opt.attSize,
                                           dec_hidden_size=4096, useS=True)
        acc = zsl_cls_VS.acc
        if best_zsl_acc_VS < acc:
            best_zsl_acc_VS = acc
            best_acc_zsl_list_VS = zsl_cls_VS.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_VS")
        log_record = 'ZSL (VS): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')
        del zsl_cls_VS

        # Train ZSL classifier in VS
        zsl_cls_VS = classifier.CLASSIFIER(syn_feature_pro, util.map_label(syn_label_pro, data.unseenclasses), \
                                           data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                           opt.syn_num, cls_mode="ZSL", netDec=zerodiff.netDec, dec_size=opt.attSize,
                                           dec_hidden_size=4096, useS=True)

        acc = zsl_cls_VS.acc
        if best_zsl_acc_VS < acc:
            best_zsl_acc_VS = acc
            best_acc_zsl_list_VS = zsl_cls_VS.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_VS")
        log_record = 'ZSL pro (VS): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')
        del zsl_cls_VS

        # Train ZSL classifier in C
        zsl_cls_C = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                                          data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                          opt.syn_num, cls_mode="ZSL", useV=False, con_size=2048, _train_C = syn_con, useC=True)
        acc = zsl_cls_C.acc
        if best_zsl_acc_C < acc:
            best_zsl_acc_C = acc
            best_acc_zsl_list_C = zsl_cls_C.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_C")
        log_record = 'ZSL (C): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')
        del zsl_cls_C

        # Train ZSL classifier in C
        zsl_cls_C = classifier.CLASSIFIER(syn_feature_pro, util.map_label(syn_label_pro, data.unseenclasses), \
                                          data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                          opt.syn_num, cls_mode="ZSL", useV=False, con_size=2048, _train_C = syn_con_pro,  useC=True)
        acc = zsl_cls_C.acc
        if best_zsl_acc_C < acc:
            best_zsl_acc_C = acc
            best_acc_zsl_list_C = zsl_cls_C.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_C")
        log_record = 'ZSL pro (C): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')
        del zsl_cls_C

        # Train ZSL classifier in VC
        zsl_cls_VC = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                                           data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                           opt.syn_num, cls_mode="ZSL", con_size=2048, _train_C = syn_con, useC=True)
        acc = zsl_cls_VC.acc
        if best_zsl_acc_VC < acc:
            best_zsl_acc_VC = acc
            best_acc_zsl_list_VC = zsl_cls_VC.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_VC")
        log_record = 'ZSL (VC): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')
        del zsl_cls_VC

        # Train ZSL classifier in VC
        zsl_cls_VC = classifier.CLASSIFIER(syn_feature_pro, util.map_label(syn_label_pro, data.unseenclasses), \
                                           data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                           opt.syn_num, cls_mode="ZSL", con_size=2048, _train_C = syn_con_pro, useC=True)
        acc = zsl_cls_VC.acc
        if best_zsl_acc_VC < acc:
            best_zsl_acc_VC = acc
            best_acc_zsl_list_VC = zsl_cls_VC.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_VC")
        log_record = 'ZSL pro (VC): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')
        del zsl_cls_VC

        # Train ZSL classifier in VCS
        zsl_cls_VCS = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                                            data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                            opt.syn_num, cls_mode="ZSL", netDec=zerodiff.netDec, dec_size=opt.attSize,
                                            dec_hidden_size=4096, useS=True, con_size=2048, _train_C = syn_con,  useC=True)
        acc = zsl_cls_VCS.acc
        if best_zsl_acc_VCS < acc:
            best_zsl_acc_VCS = acc
            best_acc_zsl_list_VCS = zsl_cls_VCS.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_VCS")
        log_record = 'ZSL (VCS): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')
        del zsl_cls_VCS

        # Train ZSL classifier in VC
        zsl_cls_VCS = classifier.CLASSIFIER(syn_feature_pro, util.map_label(syn_label_pro, data.unseenclasses), \
                                            data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                            opt.syn_num, cls_mode="ZSL", netDec=zerodiff.netDec, dec_size=opt.attSize,
                                            dec_hidden_size=4096, useS=True, con_size=2048, _train_C = syn_con_pro,
                                            useC=True)
        acc = zsl_cls_VCS.acc
        if best_zsl_acc_VCS < acc:
            best_zsl_acc_VCS = acc
            best_acc_zsl_list_VCS = zsl_cls_VCS.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_VCS")
        log_record = 'ZSL pro (VCS): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')
        del zsl_cls_VCS

        # Train Seen classifier in V
        seen_cls_V = classifier.CLASSIFIER(syn_feature_seen, util.map_label(syn_label_seen, data.seenclasses), \
                                           data, data.seenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                           opt.syn_num, cls_mode="seen")
        acc = seen_cls_V.acc
        if best_seen_acc_V < acc:
            best_seen_acc_V = acc
        log_record = 'Seen (V): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')
        del seen_cls_V

        # reset G to training mode
        zerodiff.train()

        if opt.gzsl:
            log_record = "best GZSL (V): U: %.4f, S: %.4f, H: %.4f" % \
                         (best_acc_unseen_V, best_acc_seen_V, best_gzsl_acc_V)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_seen_list (V): " + str(best_acc_seen_list_V)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_unseen_list (V): " + str(best_acc_unseen_list_V)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best GZSL (VS): U: %.4f, S: %.4f, H: %.4f" % \
                         (best_acc_unseen_VS, best_acc_seen_VS, best_gzsl_acc_VS)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_seen_list (VS): " + str(best_acc_seen_list_VS)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_unseen_list (VS): " + str(best_acc_unseen_list_VS)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best GZSL (C): U: %.4f, S: %.4f, H: %.4f" % \
                         (best_acc_unseen_C, best_acc_seen_C, best_gzsl_acc_C)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_seen_list (C): " + str(best_acc_seen_list_C)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_unseen_list (C): " + str(best_acc_unseen_list_C)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best GZSL (VC): U: %.4f, S: %.4f, H: %.4f" % \
                         (best_acc_unseen_VC, best_acc_seen_VC, best_gzsl_acc_VC)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_seen_list (VC): " + str(best_acc_seen_list_VC)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_unseen_list (VC): " + str(best_acc_unseen_list_VC)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best GZSL (VCS): U: %.4f, S: %.4f, H: %.4f" % \
                         (best_acc_unseen_VCS, best_acc_seen_VCS, best_gzsl_acc_VCS)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_seen_list (VCS): " + str(best_acc_seen_list_VCS)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_unseen_list (VCS): " + str(best_acc_unseen_list_VCS)
            print(log_record)
            logger.write(log_record + '\n')

        log_record = 'best ZSL (V): %.4f' % (best_zsl_acc_V.item())
        print(log_record)
        logger.write(log_record + '\n')

        log_record = "best_acc_zsl_list (V): " + str(best_acc_zsl_list_V)
        print(log_record)
        logger.write(log_record + '\n')

        log_record = 'best ZSL (VS): %.4f' % (best_zsl_acc_VS.item())
        print(log_record)
        logger.write(log_record + '\n')

        log_record = "best_acc_zsl_list (VS): " + str(best_acc_zsl_list_VS)
        print(log_record)
        logger.write(log_record + '\n')

        log_record = 'best ZSL (C): %.4f' % (best_zsl_acc_C.item())
        print(log_record)
        logger.write(log_record + '\n')

        log_record = "best_acc_zsl_list (C): " + str(best_acc_zsl_list_C)
        print(log_record)
        logger.write(log_record + '\n')

        log_record = 'best ZSL (VC): %.4f' % (best_zsl_acc_VC.item())
        print(log_record)
        logger.write(log_record + '\n')

        log_record = "best_acc_zsl_list (VC): " + str(best_acc_zsl_list_VC)
        print(log_record)
        logger.write(log_record + '\n')

        log_record = 'best ZSL (VCS): %.4f' % (best_zsl_acc_VCS.item())
        print(log_record)
        logger.write(log_record + '\n')

        log_record = "best_acc_zsl_list (VCS): " + str(best_acc_zsl_list_VCS)
        print(log_record)
        logger.write(log_record + '\n')

        log_record = 'best seen (V): %.4f' % (best_seen_acc_V.item())
        print(log_record)
        logger.write(log_record + '\n')

        # Classifier instances retain materialized V/S/C feature matrices and
        # CUDA optimizers. Release every evaluation artifact before returning
        # to generator training to prevent host/GPU memory growth across epochs.
        for evaluation_name in (
            'seen_cls_V',
            'gzsl_cls_V', 'gzsl_cls_VS', 'gzsl_cls_C', 'gzsl_cls_VC', 'gzsl_cls_VCS',
            'zsl_cls_V', 'zsl_cls_VS', 'zsl_cls_C', 'zsl_cls_VC', 'zsl_cls_VCS',
            'train_X', 'train_C', 'train_Y', 'train_X_pro', 'train_C_pro', 'train_Y_pro',
            'syn_feature', 'syn_con', 'syn_label',
            'syn_feature_pro', 'syn_con_pro', 'syn_label_pro',
            'syn_feature_seen', 'syn_con_seen', 'syn_label_seen',
        ):
            globals().pop(evaluation_name, None)
        gc.collect()
        if opt.cuda:
            torch.cuda.empty_cache()

    if (
        opt.training_checkpoint_interval > 0
        and (epoch % opt.training_checkpoint_interval == 0 or epoch == opt.nepoch - 1)
    ):
        checkpoint_path = save_training_state(zerodiff, model_save_name, epoch)
        checkpoint_record = 'Saved recoverable training state: ' + checkpoint_path
        print(checkpoint_record)
        logger.write(checkpoint_record + '\n')



