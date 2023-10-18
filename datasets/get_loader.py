import torch
from datasets.loader_avs import S4Dataset, MS3Dataset


def get_loader(cfg):
    if cfg.TRAIN.TASK.lower() == "s4":
        train_dataset = S4Dataset(cfg=cfg, split='train')
        val_dataset = S4Dataset(cfg=cfg, split='val')
    elif cfg.TRAIN.TASK.lower() == "ms3":
        train_dataset = MS3Dataset(cfg=cfg, split='train')
        val_dataset = MS3Dataset(cfg=cfg, split='val')
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.PARAM.BATCH_SIZE, shuffle=True,
                                                   num_workers=cfg.PARAM.NUM_WORKERS, pin_memory=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=cfg.PARAM.NUM_WORKERS, pin_memory=False)

    return train_dataloader, val_dataloader
