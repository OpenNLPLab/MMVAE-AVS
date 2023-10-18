import os
import time
import torch
import logging

from config import cfg
from torchvggish.vggish import VGGish

from utils import pyutils
from utils.utility import logger, mask_iou, save_mask, save_mask_ms3
from utils.system import setup_logging
from trainer import train_one_epoch
from datasets.get_loader import get_loader
from torch.optim import lr_scheduler
from model.AVSModel import AVSModel, AVSMMVAE
import h5py


class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea


if __name__ == "__main__":
    pyutils.set_seed(cfg.PARAM.SEED)

    # Log directory
    # if not os.path.exists(cfg.TRAIN.LOG_DIR):
    #     os.makedirs(cfg.TRAIN.LOG_DIR, exist_ok=True)
    # Logs
    # log_dir = os.path.join(cfg.TRAIN.LOG_DIR, '{}'.format(time.strftime(cfg.TRAIN.TASK + '_%Y%m%d-%H%M%S')))
    # script_path = os.path.join(log_dir, 'scripts')
    # checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    # log_path = os.path.join(log_dir, 'log')

    # Save scripts
    # if not os.path.exists(script_path): os.makedirs(script_path, exist_ok=True)
    # # Checkpoints directory
    # if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir, exist_ok=True)
    # # Set logger
    # if not os.path.exists(log_path): os.makedirs(log_path, exist_ok=True)

    # setup_logging(filename=os.path.join(log_path, 'log.txt'))
    save_mask_path = os.path.join(cfg.TRAIN.LOG_DIR, cfg.MODEL.TRAINED.split("/")[-3], "h5py")
    if not os.path.exists(save_mask_path): os.makedirs(save_mask_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))

    # Model
    model = AVSMMVAE(config=cfg)
    model = torch.nn.DataParallel(model).cuda().train()
    state_dict = torch.load(cfg.MODEL.TRAINED)
    model.module.load_state_dict(state_dict)
    # import pdb; pdb.set_trace()
    param_count = sum(x.numel()/1e6 for x in model.parameters())
    logger.info("Model have {:.4f}Mb paramerters in total".format(param_count))

    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device).cuda().eval()

    # Data
    train_dataloader, val_dataloader = get_loader(cfg)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), cfg.PARAM.LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.PARAM.LR_DECAY_STEP, gamma=cfg.PARAM.LR_DECAY_RATE)
    avg_meter_miou = pyutils.AverageMeter('miou')
    model.eval()
    print(len(val_dataloader))
    with torch.no_grad():
        for n_iter, batch_data in enumerate(val_dataloader):
            imgs, audio, mask, name_list = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

            imgs, audio, mask = imgs.cuda(), audio.cuda(), mask.cuda()
            B, T, C, H, W = imgs.shape
            imgs = imgs.view(B*T, C, H, W)
            mask = mask.view(B*T, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
            with torch.no_grad():
                audio_feature = audio_backbone(audio)
            
            output, latent_code = model(imgs, audio_feature) # [bs*5, 1, 224, 224]

            s_a, s_v, c_va = latent_code
            hf_s_a = h5py.File(os.path.join(save_mask_path, "s_a_{}.h5".format(n_iter)), 'w')
            hf_s_a.create_dataset('s_a', data=s_a.cpu().data.numpy())
            hf_s_a.close()
            hf_s_v = h5py.File(os.path.join(save_mask_path, "s_v_{}.h5".format(n_iter)), 'w')
            hf_s_v.create_dataset('s_v', data=s_v.cpu().data.numpy())
            hf_s_v.close()
            hf_c_va = h5py.File(os.path.join(save_mask_path, "c_va_{}.h5".format(n_iter)), 'w')
            hf_c_va.create_dataset('c_va', data=c_va.cpu().data.numpy())
            hf_c_va.close()
            
            if cfg.TRAIN.TASK == "S4":
                save_mask(output, save_mask_path, name_list[0], name_list[1])
            elif cfg.TRAIN.TASK == "S3":
                save_mask_ms3(output, save_mask_path, name_list[0])

            miou = mask_iou(output.squeeze(1), mask)
            avg_meter_miou.add({'miou': miou})

        miou = (avg_meter_miou.pop('miou'))
    print(miou)
