import torch
from tqdm import tqdm
from utils import pyutils
from loss import IouSemanticAwareLoss


def train_one_epoch(cfg, epoch, train_dataloader, audio_backbone, model, optimizer):
    avg_total_loss, avg_iou_loss, avg_sa_loss = pyutils.AvgMeter(), pyutils.AvgMeter(), pyutils.AvgMeter()
    progress_bar = tqdm(train_dataloader, ncols=100, desc='Epoch[{:03d}/{:03d}]'.format(epoch+1, cfg.PARAM.EPOCHS))
    for n_iter, batch_data in enumerate(progress_bar):
        optimizer.zero_grad()
        imgs, audio, mask, _ = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]

        imgs, audio, mask = imgs.cuda(), audio.cuda(), mask.cuda()
        B, T, C, H, W = imgs.shape
        imgs = imgs.view(B*T, C, H, W)
        if cfg.TRAIN.TASK == "MS3":
            mask = mask.view(B*T, 1, H, W)
        else:
            mask = mask.view(B, 1, H, W)
        audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])  # [B*T, 1, 96, 64]
        with torch.no_grad():
            audio_feature = audio_backbone(audio) # [B*T, 128]

        output, visual_feat_list, audio_feat_list, latent_loss = model(imgs, audio_feature, mask) # [bs*5, 1, 224, 224]
        loss, loss_dict = IouSemanticAwareLoss(output, mask, audio_feat_list, visual_feat_list, 
                                                lambda_1=cfg.PARAM.LAMBDA_1, 
                                                mask_pooling_type=cfg.PARAM.MASK_POOLING_TYPE)
        loss = loss + cfg.PARAM.MI_LOSS_RATE*latent_loss
        avg_total_loss.update(loss.data)
        # avg_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
        # avg_sa_loss.add({'sa_loss': loss_dict['sa_loss']})

        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=f"{avg_total_loss.show():.3f}")

    return model
