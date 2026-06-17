# author: ZihanYe
# ZeroDiff (ICLR25)
from __future__ import print_function
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
# import functions
import datasets.image_util as util
import classifiers.classifier_images as classifier
from config_zerodiff import opt
import zerodiff_tools
import torch.nn.functional as F
import os

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename + '.log', "w")
        f.close()

    def write(self, message):
        f = open(self.filename + '.log', "a")
        f.write(message)
        f.close()

for folder in ("./log", "./out", f"./log/{opt.dataset}", f"./out/{opt.dataset}"):
    os.makedirs(folder, exist_ok=True)

logger_name = "./log/%s/train_zerodiff_DFG_%dpercent_att:%s_b:%d_lr:%s_n_T:%d_betas:%s,%s_gamma:ADV:%.1f_VAE:%.1f_x0:%.1f_xt:%.1f_dist:%.1f_rel:%.1f_rd:%.1f_ra:%.1f_ang:%d_f:%.1f_rc:%.1f_rp:%d_num:%s" % (
    opt.dataset, opt.split_percent, opt.class_embedding, opt.batch_size, str(opt.lr), opt.n_T, str(opt.ddpmbeta1),
    str(opt.ddpmbeta2), opt.gamma_ADV, opt.gamma_VAE, opt.gamma_x0, opt.gamma_xt, opt.gamma_dist, opt.gamma_rel,
    opt.rel_dist_ratio, opt.rel_angle_ratio, int(opt.rel_use_angle), opt.factor_dist, opt.rel_con_weight, opt.rel_proj_dim, opt.syn_num)
logger = Logger(logger_name)
model_save_name = "./out/%s/zerodiff_DFG_%dpercent_att:%s_b:%d_lr:%s_n_T:%d_betas:%s,%s_gamma:ADV:%.1f_VAE:%.1f_x0:%.1f_xt:%.1f_dist:%.1f_rel:%.1f_rd:%.1f_ra:%.1f_ang:%d_f:%.1f_rc:%.1f_rp:%d_num:%d" % (
    opt.dataset, opt.split_percent, opt.class_embedding, opt.batch_size, str(opt.lr), opt.n_T, str(opt.ddpmbeta1),
    str(opt.ddpmbeta2), opt.gamma_ADV, opt.gamma_VAE, opt.gamma_x0, opt.gamma_xt, opt.gamma_dist, opt.gamma_rel,
    opt.rel_dist_ratio, opt.rel_angle_ratio, int(opt.rel_use_angle), opt.factor_dist, opt.rel_con_weight, opt.rel_proj_dim, opt.syn_num)

logger.write(
    "VSRA teacher weights | sem: %.2f con: %.2f | proj_dim: %d\n" % (
        opt.rel_sem_weight,
        opt.rel_con_weight,
        opt.rel_proj_dim,
    )
)


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

###########
# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_con = torch.FloatTensor(opt.batch_size, 2048)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)  # attSize class-embedding size
input_label = torch.LongTensor(opt.batch_size)  # attSize class-embedding size
input_test_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_test_con = torch.FloatTensor(opt.batch_size, 2048)
input_test_att = torch.FloatTensor(opt.batch_size, opt.attSize)
##########
# Cuda
if opt.cuda:
    input_res = input_res.cuda()
    input_att = input_att.cuda()
    input_label = input_label.cuda()
    input_con = input_con.cuda()
    input_test_res, input_test_con, input_test_att = input_test_res.cuda(), input_test_con.cuda(), input_test_att.cuda()


def loss_fn(recon_x, x, mean, log_var):
    Recon = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), reduction='sum')
    Recon = Recon.sum() / x.size(0)
    # Recon = torch.nn.functional.mse_loss(recon_x, x.detach(), reduction='sum')
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

def sampleTestSeen():
    batch_feature, batch_con, batch_att, _ = data.next_test_seen_batch(opt.batch_size)
    input_test_res.copy_(batch_feature)
    input_test_con.copy_(batch_con)
    input_test_att.copy_(batch_att)

    return input_test_res, input_test_con, input_test_att

def WeightedL14att(pred, gt):
    wt = (pred - gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
    loss = wt * (pred - gt).abs()
    return loss.sum() / loss.size(0)


def pdist(features, squared=False, eps=1e-12):
    feature_square = features.pow(2).sum(dim=1)
    product = features @ features.t()
    distances = (feature_square.unsqueeze(1) + feature_square.unsqueeze(0) - 2 * product).clamp(min=eps)
    if not squared:
        distances = distances.sqrt()

    distances = distances.clone()
    distances[range(len(features)), range(len(features))] = 0
    return distances


def rkd_distance_loss(student_features, teacher_features, eps):
    with torch.no_grad():
        teacher_distances = pdist(teacher_features, squared=False, eps=eps)
        teacher_positive = teacher_distances[teacher_distances > 0]
        teacher_mean = teacher_positive.mean() if teacher_positive.numel() > 0 else teacher_distances.new_tensor(1.0)
        teacher_distances = teacher_distances / teacher_mean.clamp_min(eps)

    student_distances = pdist(student_features, squared=False, eps=eps)
    student_positive = student_distances[student_distances > 0]
    student_mean = student_positive.mean() if student_positive.numel() > 0 else student_distances.new_tensor(1.0)
    student_distances = student_distances / student_mean.clamp_min(eps)

    return F.smooth_l1_loss(student_distances, teacher_distances, reduction='mean')


def rkd_angle_loss(student_features, teacher_features, eps, max_samples):
    n_sample = student_features.shape[0]
    if max_samples > 0 and n_sample > max_samples:
        sample_indices = torch.randperm(n_sample, device=student_features.device)[:max_samples]
        student_features = student_features[sample_indices]
        teacher_features = teacher_features[sample_indices]

    with torch.no_grad():
        teacher_diffs = teacher_features.unsqueeze(0) - teacher_features.unsqueeze(1)
        teacher_diffs = F.normalize(teacher_diffs, p=2, dim=2, eps=eps)
        teacher_angles = torch.bmm(teacher_diffs, teacher_diffs.transpose(1, 2)).reshape(-1)

    student_diffs = student_features.unsqueeze(0) - student_features.unsqueeze(1)
    student_diffs = F.normalize(student_diffs, p=2, dim=2, eps=eps)
    student_angles = torch.bmm(student_diffs, student_diffs.transpose(1, 2)).reshape(-1)

    return F.smooth_l1_loss(student_angles, teacher_angles, reduction='mean')


def get_train_steps_per_epoch(data_loader):
    return max(1, (data_loader.ntrain + opt.batch_size - 1) // opt.batch_size)

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
    torch.save({'state_dict_E': zerodiff.netE.state_dict(),
                'state_dict_G': zerodiff.netG.state_dict(),
                'state_dict_Dec': zerodiff.netDec.state_dict(),
                'state_dict_VSRARelHead': zerodiff.netVSRARelHead.state_dict(),
                'state_dict_VSRACHead': zerodiff.netVSRACHead.state_dict(),
                'state_dict_D_x0': zerodiff.netD_x0.state_dict(),
                'state_dict_D_xt': zerodiff.netD_xt.state_dict(),
                'state_dict_D_xc': zerodiff.netD_xc.state_dict(),
                'lambda1': zerodiff.lambda1,
                'checkpoint_type': 'dfg_weights',
                'save_postfix': post,
                }, save_name + post + '.tar')


def load_zerodiff(zerodiff, checkpoint_path, load_optimizers=True):
    checkpoint = torch.load(checkpoint_path, map_location=zerodiff.device)

    encoder_state = checkpoint.get('state_dict_E')
    if encoder_state is not None:
        zerodiff.netE.load_state_dict(encoder_state)

    zerodiff.netG.load_state_dict(checkpoint['state_dict_G'])
    zerodiff.netDec.load_state_dict(checkpoint['state_dict_Dec'])
    zerodiff.netD_x0.load_state_dict(checkpoint['state_dict_D_x0'])
    zerodiff.netD_xt.load_state_dict(checkpoint['state_dict_D_xt'])
    zerodiff.netD_xc.load_state_dict(checkpoint['state_dict_D_xc'])

    vsra_rel_head_state = checkpoint.get('state_dict_VSRARelHead')
    if vsra_rel_head_state is None:
        vsra_rel_head_state = checkpoint.get('state_dict_RelProj')
    if vsra_rel_head_state is not None:
        zerodiff.netVSRARelHead.load_state_dict(vsra_rel_head_state)

    vsra_c_head_state = checkpoint.get('state_dict_VSRACHead')
    if vsra_c_head_state is None:
        vsra_c_head_state = checkpoint.get('state_dict_CTeacherEmbed')
    if vsra_c_head_state is not None:
        zerodiff.netVSRACHead.load_state_dict(vsra_c_head_state)

    if load_optimizers:
        optimizer_pairs = (
            ('optimizer_E', zerodiff.optimizerE),
            ('optimizer_G', zerodiff.optimizerG),
            ('optimizer_Dec', zerodiff.optimizerDec),
            ('optimizer_VSRARelHead', zerodiff.optimizerVSRARelHead),
            ('optimizer_VSRACHead', zerodiff.optimizerVSRACHead),
            ('optimizer_D_x0', zerodiff.optimizerD_x0),
            ('optimizer_D_xt', zerodiff.optimizerD_xt),
            ('optimizer_D_xc', zerodiff.optimizerD_xc),
        )
        for key, optimizer in optimizer_pairs:
            state = checkpoint.get(key)
            if state is None and key == 'optimizer_VSRARelHead':
                state = checkpoint.get('optimizer_RelProj')
            if state is None and key == 'optimizer_VSRACHead':
                state = checkpoint.get('optimizer_CTeacherEmbed')
            if state is not None:
                optimizer.load_state_dict(state)

    lambda1 = checkpoint.get('lambda1')
    if lambda1 is not None:
        zerodiff.lambda1 = lambda1

    return checkpoint


MODALITY_ORDER = ("V", "VS", "C", "VC", "VCS")


def log_message(message):
    print(message)
    logger.write(message + '\n')


def as_scalar(value):
    return value.item() if torch.is_tensor(value) else value


def init_best_eval_state():
    return {
        modality: {
            'gzsl': {
                'seen': 0.0,
                'unseen': 0.0,
                'H': 0.0,
                'seen_list': [],
                'unseen_list': [],
            },
            'zsl': {
                'acc': 0.0,
                'acc_list': [],
            },
        }
        for modality in MODALITY_ORDER
    }


def get_eval_modality_configs(zerodiff):
    decoder_kwargs = {
        'netDec': zerodiff.netDec,
        'dec_size': opt.attSize,
        'dec_hidden_size': 4096,
    }
    return {
        'V': {
            'classifier_kwargs': {},
        },
        'VS': {
            'classifier_kwargs': {
                **decoder_kwargs,
                'useS': True,
            },
        },
        'C': {
            'classifier_kwargs': {
                'useV': False,
                'useC': True,
                'con_size': 2048,
            },
        },
        'VC': {
            'classifier_kwargs': {
                'useC': True,
                'con_size': 2048,
            },
        },
        'VCS': {
            'classifier_kwargs': {
                **decoder_kwargs,
                'useS': True,
                'useC': True,
                'con_size': 2048,
            },
        },
    }


def build_classifier_kwargs(modality_config, train_con=None):
    classifier_kwargs = dict(modality_config['classifier_kwargs'])
    if classifier_kwargs.get('useC'):
        classifier_kwargs['_train_C'] = train_con
    return classifier_kwargs


def run_classifier(train_feature, train_label, data_loader, nclass, cls_mode, modality_config, train_con=None):
    classifier_kwargs = build_classifier_kwargs(modality_config, train_con=train_con)
    return classifier.CLASSIFIER(
        train_feature,
        train_label,
        data_loader,
        nclass,
        opt.cuda,
        opt.classifier_lr,
        0.5,
        25,
        opt.syn_num,
        cls_mode=cls_mode,
        **classifier_kwargs,
    )


def update_best_gzsl(best_eval_state, modality_name, cls_result):
    best_metrics = best_eval_state[modality_name]['gzsl']
    if best_metrics['H'] < cls_result.H:
        best_metrics['seen'] = cls_result.acc_seen
        best_metrics['unseen'] = cls_result.acc_unseen
        best_metrics['H'] = cls_result.H
        best_metrics['seen_list'] = cls_result.best_acc_S_list
        best_metrics['unseen_list'] = cls_result.best_acc_U_list


def update_best_zsl(best_eval_state, modality_name, cls_result):
    best_metrics = best_eval_state[modality_name]['zsl']
    if best_metrics['acc'] < cls_result.acc:
        best_metrics['acc'] = cls_result.acc
        best_metrics['acc_list'] = cls_result.best_acc_zsl_list


def log_gzsl_result(prefix, cls_result):
    log_message('%s: U: %.4f, S: %.4f, H: %.4f' % (
        prefix,
        as_scalar(cls_result.acc_unseen),
        as_scalar(cls_result.acc_seen),
        as_scalar(cls_result.H),
    ))


def log_zsl_result(prefix, acc):
    log_message('%s: %.4f' % (prefix, as_scalar(acc)))


def build_eval_variants(data_loader, syn_feature, syn_con, syn_label, syn_feature_pro, syn_con_pro, syn_label_pro):
    eval_variants = [
        {
            'log_suffix': '',
            'syn_feature': syn_feature,
            'syn_con': syn_con,
            'syn_label': syn_label,
            'is_progressive': False,
        },
        {
            'log_suffix': ' pro',
            'syn_feature': syn_feature_pro,
            'syn_con': syn_con_pro,
            'syn_label': syn_label_pro,
            'is_progressive': True,
        },
    ]

    for eval_variant in eval_variants:
        eval_variant['train_X'] = None
        eval_variant['train_C'] = None
        eval_variant['train_Y'] = None
        if opt.gzsl:
            eval_variant['train_X'] = torch.cat((data_loader.train_feature, eval_variant['syn_feature']), 0)
            eval_variant['train_C'] = torch.cat((data_loader.train_paco, eval_variant['syn_con']), 0)
            eval_variant['train_Y'] = torch.cat((data_loader.train_label, eval_variant['syn_label']), 0)

    return eval_variants


def log_best_eval_summary(best_eval_state, best_seen_acc_v, best_main_gzsl_h):
    if opt.gzsl:
        for modality_name in MODALITY_ORDER:
            gzsl_metrics = best_eval_state[modality_name]['gzsl']
            log_message('best GZSL (%s): U: %.4f, S: %.4f, H: %.4f' % (
                modality_name,
                as_scalar(gzsl_metrics['unseen']),
                as_scalar(gzsl_metrics['seen']),
                as_scalar(gzsl_metrics['H']),
            ))
            log_message('best_acc_seen_list (%s): %s' % (modality_name, gzsl_metrics['seen_list']))
            log_message('best_acc_unseen_list (%s): %s' % (modality_name, gzsl_metrics['unseen_list']))

    for modality_name in MODALITY_ORDER:
        zsl_metrics = best_eval_state[modality_name]['zsl']
        log_message('best ZSL (%s): %.4f' % (modality_name, as_scalar(zsl_metrics['acc'])))
        log_message('best_acc_zsl_list (%s): %s' % (modality_name, zsl_metrics['acc_list']))

    log_message('best seen (V): %.4f' % as_scalar(best_seen_acc_v))
    if opt.gzsl:
        log_message('best saved DFG checkpoint (GZSL pro VCS): %.4f' % as_scalar(best_main_gzsl_h))


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
        self.netVSRARelHead = zerodiff_tools.RelationProjector(opt.resSize, opt.rel_proj_dim).to(self.device)
        self.netVSRACHead = zerodiff_tools.RelationProjector(2048, opt.rel_proj_dim).to(self.device)

        self.optimizerE = optim.Adam(self.netE.parameters(), lr=opt.lr)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerDec = optim.Adam(self.netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
        self.optimizerVSRARelHead = optim.Adam(self.netVSRARelHead.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerVSRACHead = optim.Adam(self.netVSRACHead.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
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
        self.rel_sem_weight = opt.rel_sem_weight
        self.rel_con_weight = opt.rel_con_weight
        self.rel_eps = opt.rel_eps
        self.rel_dist_ratio = opt.rel_dist_ratio
        self.rel_angle_ratio = opt.rel_angle_ratio
        self.rel_angle_max_samples = opt.rel_angle_max_samples
        self.rel_use_angle = opt.rel_use_angle

        self.loss_mse = torch.nn.MSELoss(reduction='none')

        self.batch_size = opt.batch_size
        self.data = data

        self.netR = zerodiff_tools.DRG_Generator(opt).to(self.device)
        netR_state_dict = torch.load(netR_model_path, map_location=self.device)
        netR_weights = netR_state_dict.get('state_dict_G_con') or netR_state_dict.get('state_dict_R')
        if netR_weights is None:
            raise KeyError("netR checkpoint must contain 'state_dict_G_con' or 'state_dict_R'.")
        self.netR.load_state_dict(netR_weights)
        self.netR.eval()

        resume_checkpoint_path = opt.netG_model_path or opt.model_path
        if resume_checkpoint_path:
            load_zerodiff(self, resume_checkpoint_path)
            print("Loaded DFG checkpoint from:", resume_checkpoint_path)

        self.interval_recorder_sum = {}
        self.init_recorder()

    def init_recorder(self):
        self.interval_recorder_sum['criticD_train_real_x0'] = 0.0
        self.interval_recorder_sum['criticD_train_real_xt'] = 0.0
        self.interval_recorder_sum['criticD_train_real_xc'] = 0.0
        self.interval_recorder_sum['criticD_train_fake_x0'] = 0.0
        self.interval_recorder_sum['criticD_train_fake_xt'] = 0.0
        self.interval_recorder_sum['criticD_train_fake_xc'] = 0.0
        self.interval_recorder_sum['criticD_test_real_x0'] = 0.0
        self.interval_recorder_sum['criticD_test_real_xt'] = 0.0
        self.interval_recorder_sum['criticD_test_real_xc'] = 0.0

    def get_vsra_teacher_features(self, att_0_real, vsra_c_teacher):
        teacher_features = []

        if self.rel_sem_weight > 0:
            teacher_features.append((self.rel_sem_weight, att_0_real.detach()))

        if self.rel_con_weight > 0 and vsra_c_teacher is not None:
            teacher_features.append((self.rel_con_weight, vsra_c_teacher.detach()))

        return teacher_features

    def compute_vsra_losses(self, student_features, att_0_real, vsra_c_teacher):
        vsra_distance_loss = torch.tensor(0.0, device=self.device)
        vsra_angle_loss = torch.tensor(0.0, device=self.device)

        for teacher_weight, teacher_features in self.get_vsra_teacher_features(att_0_real, vsra_c_teacher):
            vsra_distance_loss += teacher_weight * rkd_distance_loss(student_features, teacher_features, self.rel_eps)
            if self.rel_use_angle:
                vsra_angle_loss += teacher_weight * rkd_angle_loss(
                    student_features,
                    teacher_features,
                    self.rel_eps,
                    self.rel_angle_max_samples,
                )

        vsra_distance_loss = self.rel_dist_ratio * vsra_distance_loss
        vsra_angle_loss = self.rel_angle_ratio * vsra_angle_loss
        vsra_loss = vsra_distance_loss + vsra_angle_loss
        return vsra_distance_loss, vsra_angle_loss, vsra_loss

    def compute_semantic_anchor_losses(self, student_features, att_0_real):
        anchor_distance_loss = rkd_distance_loss(student_features, att_0_real.detach(), self.rel_eps)
        anchor_angle_loss = torch.tensor(0.0, device=self.device)
        if self.rel_use_angle:
            anchor_angle_loss = rkd_angle_loss(
                student_features,
                att_0_real.detach(),
                self.rel_eps,
                self.rel_angle_max_samples,
            )

        anchor_distance_loss = self.rel_dist_ratio * anchor_distance_loss
        anchor_angle_loss = self.rel_angle_ratio * anchor_angle_loss
        anchor_loss = anchor_distance_loss + anchor_angle_loss
        return anchor_distance_loss, anchor_angle_loss, anchor_loss

    def build_train_vsra_c_teacher(self, con_0_real):
        return self.netVSRACHead(con_0_real)

    def update_relation_embedding(self, x_0_real, att_0_real, con_0_real):
        real_vsra_distance_loss = torch.tensor(0.0, device=self.device)
        real_vsra_angle_loss = torch.tensor(0.0, device=self.device)
        real_vsra_loss = torch.tensor(0.0, device=self.device)
        vsra_c_teacher = None
        if self.gamma_rel <= 0:
            return real_vsra_distance_loss, real_vsra_angle_loss, real_vsra_loss, vsra_c_teacher

        for p in self.netVSRARelHead.parameters():
            p.requires_grad = True
        for p in self.netVSRACHead.parameters():
            p.requires_grad = True

        self.netVSRARelHead.zero_grad()
        self.netVSRACHead.zero_grad()
        self.optimizerVSRARelHead.zero_grad()
        self.optimizerVSRACHead.zero_grad()

        vsra_c_teacher = self.build_train_vsra_c_teacher(con_0_real)
        vsra_student_real = self.netVSRARelHead(x_0_real)
        real_vsra_distance_loss, real_vsra_angle_loss, real_vsra_loss = self.compute_vsra_losses(
            vsra_student_real,
            att_0_real,
            vsra_c_teacher,
        )
        c_anchor_distance_loss, c_anchor_angle_loss, _ = self.compute_semantic_anchor_losses(vsra_c_teacher, att_0_real)
        real_vsra_distance_loss = real_vsra_distance_loss + c_anchor_distance_loss
        real_vsra_angle_loss = real_vsra_angle_loss + c_anchor_angle_loss
        real_vsra_loss = real_vsra_distance_loss + real_vsra_angle_loss
        (self.gamma_rel * real_vsra_loss).backward()
        self.optimizerVSRARelHead.step()
        self.optimizerVSRACHead.step()

        with torch.no_grad():
            vsra_c_teacher = self.build_train_vsra_c_teacher(con_0_real).detach()
        return real_vsra_distance_loss, real_vsra_angle_loss, real_vsra_loss, vsra_c_teacher

    def compute_generator_vsra_losses(self, x_0_fake, att_0_real, vsra_c_teacher):
        vsra_distance_loss = torch.tensor(0.0, device=self.device)
        vsra_angle_loss = torch.tensor(0.0, device=self.device)
        vsra_loss = torch.tensor(0.0, device=self.device)
        if self.gamma_rel <= 0:
            return vsra_distance_loss, vsra_angle_loss, vsra_loss

        vsra_student_fake = self.netVSRARelHead(x_0_fake)
        return self.compute_vsra_losses(vsra_student_fake, att_0_real, vsra_c_teacher)

    def forward(self):
        gp_sum = 0  # Running sum used for adaptive lambda scaling.
        for _ in range(opt.critic_iter):
            x_0_real, con_0_real, att_0_real, _ = sample(self.batch_size)
            D_cost, Wasserstein_D, gp_sum, distill_loss = self.update_D(x_0_real, con_0_real, att_0_real, gp_sum)

        gp_sum /= (self.gamma_ADV * self.lambda1 * opt.critic_iter)
        if gp_sum > 1.05:
            self.lambda1 *= 1.1
        elif gp_sum < 1.001:
            self.lambda1 /= 1.1
        G_cost, vae_loss_seen, real_vsra_distance_loss, real_vsra_angle_loss, real_vsra_loss, vsra_distance_loss, vsra_angle_loss, vsra_loss = self.update_G(x_0_real, con_0_real, att_0_real)
        return D_cost, Wasserstein_D, distill_loss, G_cost, vae_loss_seen, real_vsra_distance_loss, real_vsra_angle_loss, real_vsra_loss, vsra_distance_loss, vsra_angle_loss, vsra_loss

    def update_D(self, x_0_real, con_0_real, att_0_real, gp_sum):
        for p in self.netE.parameters():
            p.requires_grad = False
        for p in self.netG.parameters():
            p.requires_grad = False
        for p in self.netVSRARelHead.parameters():
            p.requires_grad = False
        for p in self.netVSRACHead.parameters():
            p.requires_grad = False
        for p in self.netD_x0.parameters():
            p.requires_grad = True
        for p in self.netD_xt.parameters():
            p.requires_grad = True
        for p in self.netD_xc.parameters():
            p.requires_grad = True
        for p in self.netDec.parameters():
            p.requires_grad = True

        z, _, _ = self.netE(x_0_real, att_0_real)

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
        criticD_real_xt_mean = -self.netD_xt(x_t_real, x_tp1_real, att_0_real, con_0_real, _ts_feat).mean() if self.gamma_xt > 0 else torch.tensor(0.0).to(self.device)
        criticD_real_xc = -self.netD_xc(x_0_real, con_0_real).mean()
        criticD_real = self.gamma_x0 * criticD_real_x0 + self.gamma_xt * criticD_real_xt_mean + criticD_real_xc
        criticD_real = self.gamma_ADV * criticD_real
        criticD_real.backward()

        criticD_fake_x0 = self.netD_x0(x_0_fake.detach(), att_0_real).mean() if self.gamma_x0 > 0 else torch.tensor(0.0).to(self.device)
        criticD_fake_xt_mean = self.netD_xt(x_t_fake.detach(), x_tp1_real, att_0_real, con_0_real, _ts_feat).mean() if self.gamma_xt > 0 else torch.tensor(0.0).to(self.device)
        criticD_fake_xc = self.netD_xc(x_0_fake.detach(), con_0_real).mean()
        criticD_fake = self.gamma_x0 * criticD_fake_x0 + self.gamma_xt * criticD_fake_xt_mean + criticD_fake_xc
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
            test_seen_x_t_real, test_seen_x_tp1_real, _ = self.q_sample_pairs(test_seen_x_0_real, _ts_feat)
            criticD_test_real_x0 = self.netD_x0(test_seen_x_0_real, test_seen_att_0_real)
            criticD_test_real_xt = self.netD_xt(test_seen_x_t_real, test_seen_x_tp1_real, test_seen_att_0_real, test_seen_con_0_real, _ts_feat)
            criticD_test_real_xc = self.netD_xc(test_seen_x_0_real, test_seen_con_0_real)

        self.interval_recorder_sum['criticD_train_real_x0'] += criticD_real_x0.mean()
        self.interval_recorder_sum['criticD_train_real_xt'] += criticD_real_xt.mean()
        self.interval_recorder_sum['criticD_train_real_xc'] += criticD_real_xc.mean()
        self.interval_recorder_sum['criticD_train_fake_x0'] += criticD_fake_x0.mean()
        self.interval_recorder_sum['criticD_train_fake_xt'] += criticD_fake_xt.mean()
        self.interval_recorder_sum['criticD_train_fake_xc'] += criticD_fake_xc.mean()
        self.interval_recorder_sum['criticD_test_real_x0'] += criticD_test_real_x0.mean()
        self.interval_recorder_sum['criticD_test_real_xt'] += criticD_test_real_xt.mean()
        self.interval_recorder_sum['criticD_test_real_xc'] += criticD_test_real_xc.mean()

        return D_cost, Wasserstein_D, gp_sum, distill_loss

    def update_G(self, x_0_real, con_0_real, att_0_real):
        real_vsra_distance_loss = torch.tensor(0.0, device=self.device)
        real_vsra_angle_loss = torch.tensor(0.0, device=self.device)
        real_vsra_loss = torch.tensor(0.0, device=self.device)
        vsra_c_teacher = None
        if self.gamma_rel > 0:
            real_vsra_distance_loss, real_vsra_angle_loss, real_vsra_loss, vsra_c_teacher = self.update_relation_embedding(
                x_0_real,
                att_0_real,
                con_0_real,
            )

        for p in self.netE.parameters():
            p.requires_grad = True
        for p in self.netG.parameters():
            p.requires_grad = True
        for p in self.netVSRARelHead.parameters():
            p.requires_grad = False
        for p in self.netVSRACHead.parameters():
            p.requires_grad = False
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
        self.netVSRARelHead.zero_grad()

        z, means, log_var = self.netE(x_0_real, att_0_real)

        _ts_feat = torch.randint(0, self.n_T, (self.batch_size,), dtype=torch.int64).to(self.device)
        _, x_tp1_real, _ = self.q_sample_pairs(x_0_real, _ts_feat)
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

        vsra_distance_loss, vsra_angle_loss, vsra_loss = self.compute_generator_vsra_losses(
            x_0_fake,
            att_0_real,
            vsra_c_teacher,
        )
        errG += self.gamma_rel * vsra_loss

        errG.backward()
        # write a condition here
        self.optimizerE.step()
        self.optimizerG.step()
        if self.gamma_recons > 0 and not opt.freeze_dec:  # not train decoder at feedback time
            self.optimizerDec.step()
        return G_cost, vae_loss_seen, real_vsra_distance_loss, real_vsra_angle_loss, real_vsra_loss, vsra_distance_loss, vsra_angle_loss, vsra_loss

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
zerodiff.train()

modality_configs = get_eval_modality_configs(zerodiff)
best_eval_state = init_best_eval_state()
best_seen_acc_V = 0.0
best_main_gzsl_H = 0.0


n_iter = get_train_steps_per_epoch(data)
for epoch in range(0, opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):
        D_cost, Wasserstein_D, distill_loss, G_cost, vae_loss_seen, real_vsra_distance_loss, real_vsra_angle_loss, real_vsra_loss, vsra_distance_loss, vsra_angle_loss, vsra_loss = zerodiff()

    log_message('[%d/%d] Loss_D: %.4f, Wasserstein_dist:%.4f, distill_loss:%.4f' % (
        epoch, opt.nepoch, D_cost.item(), Wasserstein_D.item(), distill_loss.item()))

    log_message('[%d/%d] Loss_G: %.4f, vae_loss_seen:%.4f, real_vsra_distance_loss:%.4f, real_vsra_angle_loss:%.4f, real_vsra_loss:%.4f, vsra_distance_loss:%.4f, vsra_angle_loss:%.4f, vsra_loss:%.4f' % (
        epoch, opt.nepoch, G_cost.item(), vae_loss_seen.item(), real_vsra_distance_loss.item(), real_vsra_angle_loss.item(), real_vsra_loss.item(), vsra_distance_loss.item(), vsra_angle_loss.item(), vsra_loss.item()))

    criticD_train_real_x0 = zerodiff.interval_recorder_sum['criticD_train_real_x0'].item() / n_iter
    criticD_train_real_xt = zerodiff.interval_recorder_sum['criticD_train_real_xt'].item() / n_iter
    criticD_train_real_xc = zerodiff.interval_recorder_sum['criticD_train_real_xc'].item() / n_iter

    criticD_test_real_x0 = zerodiff.interval_recorder_sum['criticD_test_real_x0'].item() / n_iter
    criticD_test_real_xt = zerodiff.interval_recorder_sum['criticD_test_real_xt'].item() / n_iter
    criticD_test_real_xc = zerodiff.interval_recorder_sum['criticD_test_real_xc'].item() / n_iter

    criticD_train_fake_x0 = zerodiff.interval_recorder_sum['criticD_train_fake_x0'].item() / n_iter
    criticD_train_fake_xt = zerodiff.interval_recorder_sum['criticD_train_fake_xt'].item() / n_iter
    criticD_train_fake_xc = zerodiff.interval_recorder_sum['criticD_train_fake_xc'].item() / n_iter
    zerodiff.init_recorder()

    log_message('[%d/%d] D_train_real_x0: %.6f, D_train_real_xt: %.6f, D_train_real_xc: %.6f' % (
        epoch, opt.nepoch, criticD_train_real_x0, criticD_train_real_xt, criticD_train_real_xc))

    log_message('[%d/%d] D_test_real_x0: %.6f, D_test_real_xt: %.6f, D_test_real_xc: %.6f' % (
        epoch, opt.nepoch, criticD_test_real_x0, criticD_test_real_xt, criticD_test_real_xc))

    log_message('[%d/%d] D_train_fake_x0: %.6f, D_train_fake_xt: %.6f, D_train_fake_xc: %.6f' % (
        epoch, opt.nepoch, criticD_train_fake_x0, criticD_train_fake_xt, criticD_train_fake_xc))

    if epoch % opt.eval_interval == 0 or epoch == (opt.nepoch - 1):
        zerodiff.eval()
        syn_feature, syn_con, syn_label = generate_syn_feature(zerodiff, data.unseenclasses, data.attribute, opt.syn_num)
        syn_feature_pro, syn_con_pro, syn_label_pro = generate_syn_feature(zerodiff, data.unseenclasses, data.attribute, opt.syn_num, progressive=True)
        syn_feature_seen, _, syn_label_seen = generate_syn_feature(zerodiff, data.seenclasses, data.attribute, opt.syn_num)

        eval_variants = build_eval_variants(
            data,
            syn_feature,
            syn_con,
            syn_label,
            syn_feature_pro,
            syn_con_pro,
            syn_label_pro,
        )

        seen_cls_V = run_classifier(
            syn_feature_seen,
            util.map_label(syn_label_seen, data.seenclasses),
            data,
            data.seenclasses.size(0),
            "seen",
            modality_configs['V'],
        )
        if best_seen_acc_V < seen_cls_V.acc:
            best_seen_acc_V = seen_cls_V.acc
        log_zsl_result('Seen (V)', seen_cls_V.acc)

        if opt.gzsl:
            for eval_variant in eval_variants:
                for modality_name in MODALITY_ORDER:
                    modality_config = modality_configs[modality_name]
                    gzsl_cls = run_classifier(
                        eval_variant['train_X'],
                        eval_variant['train_Y'],
                        data,
                        opt.nclass_all,
                        "GZSL",
                        modality_config,
                        train_con=eval_variant['train_C'],
                    )
                    update_best_gzsl(
                        best_eval_state,
                        modality_name,
                        gzsl_cls,
                    )
                    current_gzsl_h = as_scalar(gzsl_cls.H)
                    if eval_variant['is_progressive'] and modality_name == 'VCS' and best_main_gzsl_H < current_gzsl_h:
                        best_main_gzsl_H = current_gzsl_h
                        save_zerodiff(zerodiff, model_save_name, '_gzsl')
                    log_gzsl_result('GZSL%s (%s)' % (eval_variant['log_suffix'], modality_name), gzsl_cls)

        for eval_variant in eval_variants:
            mapped_syn_label = util.map_label(eval_variant['syn_label'], data.unseenclasses)
            for modality_name in MODALITY_ORDER:
                modality_config = modality_configs[modality_name]
                zsl_cls = run_classifier(
                    eval_variant['syn_feature'],
                    mapped_syn_label,
                    data,
                    data.unseenclasses.size(0),
                    "ZSL",
                    modality_config,
                    train_con=eval_variant['syn_con'],
                )
                update_best_zsl(
                    best_eval_state,
                    modality_name,
                    zsl_cls,
                )
                log_zsl_result('ZSL%s (%s)' % (eval_variant['log_suffix'], modality_name), zsl_cls.acc)

        zerodiff.train()
        log_best_eval_summary(best_eval_state, best_seen_acc_V, best_main_gzsl_H)



