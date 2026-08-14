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
parser.add_argument("--encoder_layer_sizes", type=list, default=[8192, 4096])
parser.add_argument("--decoder_layer_sizes", type=list, default=[4096, 8192])
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
### time-aware multi-granularity relation consistency (disabled by default)
parser.add_argument("--gamma_rel", type=float, default=0.0,
                    help="overall relation-consistency weight; 0 preserves the baseline")
parser.add_argument("--rel_class_weight", type=float, default=1.0)
parser.add_argument("--rel_instance_weight", type=float, default=1.0)
parser.add_argument("--rel_temporal_weight", type=float, default=1.0)
parser.add_argument("--rel_n_way", type=int, default=8,
                    help="classes per balanced relation episode")
parser.add_argument("--rel_k_shot", type=int, default=8,
                    help="instances per class in a balanced relation episode")
parser.add_argument("--rel_time_mode", default="uniform",
                    choices=["uniform", "class_high_noise", "instance_high_noise", "custom"])
parser.add_argument("--rel_class_t0", type=float, default=1.0)
parser.add_argument("--rel_class_tT", type=float, default=1.0)
parser.add_argument("--rel_instance_t0", type=float, default=1.0)
parser.add_argument("--rel_instance_tT", type=float, default=1.0)
parser.add_argument("--rel_gradient_mode", default="base_anchor", choices=["sum", "base_anchor"])
parser.add_argument("--rel_conflict_eps", type=float, default=1e-12)
###
parser.add_argument("--embed_type",  default='V', help='V/VA')
parser.add_argument("--n_T", type=int, default=4)
parser.add_argument("--dim_t", type=int, default=2048)
parser.add_argument("--embClsSize", type=int, default=2048)
parser.add_argument("--embConSize", type=int, default=512)
parser.add_argument("--eval_interval", type=int, default=1)
parser.add_argument('--ddpmbeta1', type=float, default=1e-1)
parser.add_argument('--ddpmbeta2', type=float, default=20)
# pretrain
parser.add_argument('--init_con', action='store_true', default=False, help='enbale MinMaxScaler on contrastive representations')
parser.add_argument("--netE_con_model_path", default=None)
parser.add_argument("--netR_model_path", default=None)
# eval
parser.add_argument("--netG_model_path", default=None)
#visualize
parser.add_argument("--model_path", default=None)

opt = parser.parse_args()
opt.lambda2 = opt.lambda1
opt.encoder_layer_sizes[0] = opt.resSize
opt.decoder_layer_sizes[-1] = opt.resSize
opt.latent_size = opt.attSize
if opt.gamma_rel > 0 and opt.rel_n_way * opt.rel_k_shot != opt.batch_size:
    parser.error("relation training requires rel_n_way * rel_k_shot == batch_size")
if min(
    opt.rel_class_weight,
    opt.rel_instance_weight,
    opt.rel_temporal_weight,
    opt.rel_class_t0,
    opt.rel_class_tT,
    opt.rel_instance_t0,
    opt.rel_instance_tT,
    opt.gamma_rel,
) < 0:
    parser.error("relation loss weights must be non-negative")
if opt.rel_conflict_eps <= 0:
    parser.error("rel_conflict_eps must be positive")

