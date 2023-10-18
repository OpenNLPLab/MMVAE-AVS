from tkinter import N
from easydict import EasyDict as edict
from utils.dataset_path import get_path
from utils.pyutils import str2bool
import argparse


parser = argparse.ArgumentParser()    
parser.add_argument("--save_name", type=str, default="debug")
parser.add_argument("--task", type=str, choices=("MS3", "S4"))
parser.add_argument("--encoder", type=str, choices=("pvt", "resnet"), default="pvt")
parser.add_argument("--neck", type=str, choices=("basic", "aspp"), default="basic")
parser.add_argument('--mi_loss', type=float, default=0.001)
parser.add_argument('--diff_loss', type=float, default=0.001)
parser.add_argument('--kl_loss', type=float, default=0.01)
parser.add_argument('--sim_loss', type=float, default=0.5)
parser.add_argument('--vae_latent', type=int, default=16)
parser.add_argument('--mi_latent', type=int, default=6)
parser.add_argument('--augment', type=str2bool, default=True)
parser.add_argument('--load', type=str, default=None)
args = parser.parse_args()


"""
default config
"""
cfg = edict()

##############################
# TRAIN
cfg.TRAIN = edict()
# TRAIN.SCHEDULER
cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "pretrained_backbones/vggish-10086976.pth"
cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = False
cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "pretrained_backbones/vggish_pca_params-970ea276.pth"
cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = False
cfg.TRAIN.PRETRAINED_RESNET50_PATH = "pretrained_backbones/resnet50-19c8e357.pth"
cfg.TRAIN.PRETRAINED_PVTV2_PATH = "pretrained_backbones/pvt_v2_b5.pth"
cfg.TRAIN.PRETRAINED_SWIN_PATH = "pretrained_backbones/swin_base_patch4_window7_224.pth"
cfg.TRAIN.LOG_DIR = "./experiments"
cfg.TRAIN.SAVE_NAME = args.save_name
cfg.TRAIN.TASK = args.task  #  "S4" or "MS3"
###############################
# DATA
cfg.DATA = edict()
path_s4, path_ms3 = get_path("lab")
if cfg.TRAIN.TASK == "S4":
    cfg.DATA.ANNO_CSV = path_s4[0]
    cfg.DATA.DIR_IMG = path_s4[1]
    cfg.DATA.DIR_AUDIO_LOG_MEL = path_s4[2]
    cfg.DATA.DIR_MASK = path_s4[3]
elif cfg.TRAIN.TASK == "MS3":
    cfg.DATA.ANNO_CSV = path_ms3[0]
    cfg.DATA.DIR_IMG = path_ms3[1]
    cfg.DATA.DIR_AUDIO_LOG_MEL = path_ms3[2]
    cfg.DATA.DIR_MASK = path_ms3[3]
cfg.DATA.IMG_SIZE = (224, 224)
cfg.DATA.RANDOM_FLIP = args.augment
###############################
# MODEL
cfg.MODEL = edict()
cfg.MODEL.ENCODER = args.encoder  # resnet, pvt, swin
cfg.MODEL.NECK = args.neck   # aspp, basic
cfg.MODEL.DECODER = "simple_no_module"
cfg.MODEL.FUSION = "tpavi" # av_corr, tpavi
cfg.MODEL.NECK_CHANNEL = 128
cfg.MODEL.MUTUAL_REG_CHANNEL = 64
cfg.MODEL.MUTUAL_REG_SIZE = args.mi_latent
cfg.MODEL.VAE_LATENT_SIZE = args.vae_latent
cfg.MODEL.TRAINED = args.load
# PARAM
cfg.PARAM = edict()
cfg.PARAM.BATCH_SIZE = 4
cfg.PARAM.NUM_WORKERS = 8
cfg.PARAM.EPOCHS = 30 if cfg.TRAIN.TASK == "MS3" else 15
cfg.PARAM.LR = 0.0001
cfg.PARAM.LR_DECAY_RATE = 1
cfg.PARAM.LR_DECAY_STEP = 12
cfg.PARAM.SEED = 1234
cfg.PARAM.SA_LOSS_STAGES = [0, 1, 2, 3]
cfg.PARAM.TPAVI_STAGES = [0, 1, 2, 3]
cfg.PARAM.MASK_POOLING_TYPE = 'avg'
cfg.PARAM.TPAVI_VA_FLAGE = True
cfg.PARAM.MI_LOSS = args.mi_loss
cfg.PARAM.DIFF_LOSS = args.diff_loss
cfg.PARAM.KL_LOSS = args.kl_loss
cfg.PARAM.SIM_LOSS = args.sim_loss
# BACKUP
cfg.BACKUP = edict()
cfg.BACKUP.BASE_NAME = f"{cfg.TRAIN.TASK}_{cfg.MODEL.ENCODER}_{cfg.MODEL.NECK}"
cfg.BACKUP.PARAM_STR = f"{args.mi_loss}_{args.diff_loss}_{args.kl_loss}_{args.sim_loss}"
cfg.BACKUP.FOLDER_NAME = f"{cfg.BACKUP.BASE_NAME}_{cfg.BACKUP.PARAM_STR}"