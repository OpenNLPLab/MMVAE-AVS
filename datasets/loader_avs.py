import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from datasets.dataset_utils import load_audio_lm, load_image_in_PIL_to_Tensor, load_image_in_PIL_to_Tensor
from datasets.transforms import RandomHorizontalFlip


class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, cfg, split='train'):
        super(S4Dataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self.mask_num = 1 if self.split == 'train' else 5
        df_all = pd.read_csv(self.cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if self.cfg.DATA.RANDOM_FLIP == True:
            self.random_horizontal_flip = RandomHorizontalFlip(0.3)

    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, category = df_one_video[0], df_one_video[2]
        img_base_path = os.path.join(self.cfg.DATA.DIR_IMG, self.split, category, video_name)
        audio_lm_path = os.path.join(self.cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, category, video_name + '.pkl')
        mask_base_path = os.path.join(self.cfg.DATA.DIR_MASK, self.split, category, video_name)
        audio_log_mel = load_audio_lm(audio_lm_path)

        imgs, masks = [], []
        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id)), transform=self.img_transform)
            imgs.append(img)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='1')
            masks.append(mask)
        if self.cfg.DATA.RANDOM_FLIP == True and self.split == "train":
            img_one, mask_one = self.random_horizontal_flip(imgs[0], masks[0])
            imgs[0], masks[0] = img_one, mask_one
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        return imgs_tensor, audio_log_mel, masks_tensor, [category, video_name]

    def __len__(self):
        return len(self.df_split)


class MS3Dataset(Dataset):
    """Dataset for multiple sound source segmentation"""
    def __init__(self, cfg, split='train'):
        super(MS3Dataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self.mask_num = 5
        df_all = pd.read_csv(self.cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if self.cfg.DATA.RANDOM_FLIP == True:
            self.random_horizontal_flip = RandomHorizontalFlip(0.3)

    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name = df_one_video[0]
        img_base_path = os.path.join(self.cfg.DATA.DIR_IMG, video_name)
        audio_lm_path = os.path.join(self.cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, video_name + '.pkl')
        mask_base_path = os.path.join(self.cfg.DATA.DIR_MASK, self.split, video_name)
        audio_log_mel = load_audio_lm(audio_lm_path)
        imgs, masks = [], []
        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id)), transform=self.img_transform)
            imgs.append(img)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='P')
            masks.append(mask)
        if self.cfg.DATA.RANDOM_FLIP == True and self.split == "train":
            for i in range(len(imgs)):
                img_one, mask_one = self.random_horizontal_flip(imgs[i], masks[i])
                imgs[i], masks[i] = img_one, mask_one
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        return imgs_tensor, audio_log_mel, masks_tensor, [video_name]

    def __len__(self):
        return len(self.df_split)
