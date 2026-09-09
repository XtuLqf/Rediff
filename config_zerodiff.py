#author: ZihanYe
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--class_embedding_norm', action='store_true', default=False)
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--noiseSize', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train GANs ')
parser.add_argument('--feed_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--dec_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--encoded_noise', action='store_true', default=False, help='enables validation mode')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--validation', action='store_true', default=False, help='enables validation mode')
parser.add_argument("--encoder_layer_sizes", type=int, nargs=2, default=[8192, 4096],
                    help="two integer layer sizes, e.g. --encoder_layer_sizes 2048 4096")
parser.add_argument("--decoder_layer_sizes", type=int, nargs=2, default=[4096, 8192],
                    help="two integer layer sizes, e.g. --decoder_layer_sizes 4096 2048")
parser.add_argument("--conditional", action='store_true',default=True)
parser.add_argument("--split_percent", type=int, default=100)
###
parser.add_argument('--freeze_dec', action='store_true', default=False, help='Freeze Decoder for fake samples')
###
parser.add_argument('--gamma_ADV', type=float, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gamma_VAE', type=float, default=1.0, help='weight on the W-GAN loss')
parser.add_argument('--gamma_x0', type=float, default=1.0, help='weight on the W-GAN loss')
parser.add_argument('--gamma_xt', type=float, default=1.0, help='weight on the W-GAN loss')
parser.add_argument('--gamma_recons', type=float, default=1.0, help='recons_weight for decoder')
parser.add_argument("--gamma_dist", type=float, default=1.0)
parser.add_argument("--gamma_CON", type=float, default=0.0)
parser.add_argument("--temp_con", type=float, default=0.07)
parser.add_argument("--gamma_CON_sample", type=float, default=0.0)
parser.add_argument("--gamma_CON_step", type=float, default=0.0)
parser.add_argument("--factor_dist", type=float, default=1.0)
### time-aware VSRA (disabled by default)
parser.add_argument("--gamma_rel", type=float, default=0.0,
                    help="overall time-aware VSRA weight; 0 preserves the baseline")
parser.add_argument("--rel_class_weight", type=float, default=1.0,
                    help="semantic-teacher weight in terminal VSRA")
parser.add_argument("--rel_instance_weight", type=float, default=1.0,
                    help="contrastive-teacher weight in terminal VSRA")
parser.add_argument("--rel_proj_dim", type=int, default=512,
                    help="dimension of the calibrated VSRA relation space")
parser.add_argument("--rel_teacher_anchor_weight", type=float, default=1.0,
                    help="weight for preserving semantic and PaCo teacher topology")
parser.add_argument("--rel_dist_ratio", type=float, default=1.0)
parser.add_argument("--rel_angle_ratio", type=float, default=2.0)
parser.add_argument("--rel_angle_max_samples", type=int, default=128)
parser.add_argument("--rel_use_angle", action="store_true", default=False)
parser.add_argument("--rel_time_pair_weight", type=float, default=1.0,
                    help="convex mix from static VSRA (0) to time-aware dual topology (1)")
parser.add_argument("--rel_time_mode", default="diffusion_reliability",
                    choices=["fixed", "class_up_instance_down", "diffusion_reliability"],
                    help="relation coordination: fixed, legacy linear, or diffusion-state reliability")
parser.add_argument("--rel_time_strength", type=float, default=0.5,
                    help="diffusion sensitivity; 0 exactly recovers fixed topology weighting")
parser.add_argument("--rel_reliability_floor", type=float, default=0.5,
                    help="minimum trust retained for either topology at low SNR")
parser.add_argument("--rel_topology_norm", default="timestep",
                    choices=["global", "timestep"],
                    help="normalize relation distances globally or within each timestep")
###
parser.add_argument("--embed_type",  default='V', help='V/VA')
parser.add_argument("--n_T", type=int, default=4)
parser.add_argument("--dim_t", type=int, default=2048)
parser.add_argument("--embClsSize", type=int, default=2048)
parser.add_argument("--embConSize", type=int, default=512)
parser.add_argument("--eval_interval", type=int, default=1)
parser.add_argument('--ddpmbeta1', type=float, default=1e-1)
parser.add_argument('--ddpmbeta2', type=float, default=20)
parser.add_argument("--netR_model_path", default=None)
parser.add_argument("--resume_training", default=None,
                    help="full DFG training-state checkpoint to resume")
parser.add_argument("--training_checkpoint_interval", type=int, default=5,
                    help="save recoverable training state every N epochs; 0 disables")

opt = parser.parse_args()
opt.lambda2 = opt.lambda1
opt.encoder_layer_sizes[0] = opt.resSize
opt.decoder_layer_sizes[-1] = opt.resSize
opt.latent_size = opt.attSize
if min(
    opt.rel_class_weight,
    opt.rel_instance_weight,
    opt.rel_teacher_anchor_weight,
    opt.rel_time_pair_weight,
    opt.gamma_rel,
) < 0:
    parser.error("relation loss weights must be non-negative")
if opt.gamma_rel > 0 and opt.rel_proj_dim <= 0:
    parser.error("gamma_rel > 0 requires rel_proj_dim > 0")
if opt.rel_dist_ratio < 0 or opt.rel_angle_ratio < 0:
    parser.error("relation distance and angle ratios must be non-negative")
if opt.rel_angle_max_samples < 0:
    parser.error("rel_angle_max_samples must be non-negative")
if not 0.0 <= opt.rel_time_strength <= 1.0:
    parser.error("rel_time_strength must be in [0, 1]")
if not 0.0 <= opt.rel_reliability_floor <= 1.0:
    parser.error("rel_reliability_floor must be in [0, 1]")
if not 0.0 <= opt.rel_time_pair_weight <= 1.0:
    parser.error("rel_time_pair_weight must be in [0, 1]")
if opt.training_checkpoint_interval < 0:
    parser.error("training_checkpoint_interval must be non-negative")

