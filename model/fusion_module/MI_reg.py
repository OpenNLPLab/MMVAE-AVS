import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks.base_blocks import BasicConv2d
from torch.distributions import Normal, Independent
from utils.torch_utils import torch_L2normalize, torch_kl_div, torch_reparametrize, tile_hw_feature


class Mutual_info_reg(nn.Module):
    def __init__(self, cfg, shape):
        super(Mutual_info_reg, self).__init__()
        self.input_channels = cfg.MODEL.NECK_CHANNEL
        self.hidden_channels = cfg.MODEL.MUTUAL_REG_CHANNEL
        self.latent_size = cfg.MODEL.MUTUAL_REG_SIZE
        self.rgb_channel_reduce = BasicConv2d(self.input_channels, self.hidden_channels, kernel_size=1)
        self.audio_channel_reduce = nn.Linear(self.input_channels, self.hidden_channels)

        self.shape = shape
        self.mid_activation = nn.ReLU
        self.rgb_mu = nn.Linear(self.hidden_channels*shape[0]*shape[1], self.latent_size)
        self.rgb_logvar = nn.Linear(self.hidden_channels*shape[0]*shape[1], self.latent_size)
        self.audio_mu = nn.Linear(self.hidden_channels, self.latent_size)
        self.audio_logvar = nn.Linear(self.hidden_channels, self.latent_size)
        # self.rgb_mu = nn.Sequential(
        #     nn.Linear(self.hidden_channels*shape[0]*shape[1], self.latent_size),
        #     self.mid_activation(),
        #     nn.Linear(self.latent_size, self.latent_size)
        # )
        # self.rgb_logvar = nn.Sequential(
        #     nn.Linear(self.hidden_channels*shape[0]*shape[1], self.latent_size),
        #     self.mid_activation(),
        #     nn.Linear(self.latent_size, self.latent_size)
        # )
        # self.audio_mu = nn.Sequential(
        #     nn.Linear(self.hidden_channels, self.latent_size),
        #     self.mid_activation(),
        #     nn.Linear(self.latent_size, self.latent_size)
        # )
        # self.audio_logvar = nn.Sequential(
        #     nn.Linear(self.hidden_channels, self.latent_size),
        #     self.mid_activation(),
        #     nn.Linear(self.latent_size, self.latent_size)
        # )
        self.fuse = nn.Linear(self.latent_size*2, self.latent_size)

    def forward(self, rgb_feat, audio_feat):
        # rgb: [BT, C, H, W], audio: [B, C, T]
        BT, C, H, W = rgb_feat.shape
        audio_feat = audio_feat.permute(0, 2, 1).view(BT, C)
        rgb_feat = torch_L2normalize(rgb_feat, d=1)
        audio_feat = torch_L2normalize(audio_feat, d=1)
        rgb_feat = self.rgb_channel_reduce(rgb_feat)
        audio_feat = self.audio_channel_reduce(audio_feat)
        rgb_feat = rgb_feat.contiguous().view(-1, self.hidden_channels*H*W)

        mu_rgb = torch.tanh(self.rgb_mu(rgb_feat))
        logvar_rgb = torch.tanh(self.rgb_logvar(rgb_feat))
        mu_audio = torch.tanh(self.audio_mu(audio_feat))
        logvar_audio = torch.tanh(self.audio_logvar(audio_feat))
        
        z_rgb = torch_reparametrize(mu_rgb, logvar_rgb)
        z_audio = torch_reparametrize(mu_audio, logvar_audio)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        dist_audio = Independent(Normal(loc=mu_audio, scale=torch.exp(logvar_audio)), 1)

        bi_di_kld = torch.mean(torch_kl_div(dist_rgb, dist_audio)) + torch.mean(torch_kl_div(dist_audio, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_audio)
        ce_rgb_depth = F.binary_cross_entropy(z_rgb_norm, z_depth_norm.detach())
        ce_depth_rgb = F.binary_cross_entropy(z_depth_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_depth + ce_depth_rgb - bi_di_kld
        z_fuse = self.fuse(torch.cat([z_rgb, z_audio], dim=1))

        return {"loss": latent_loss, 
                "features": [tile_hw_feature(z_rgb, self.shape), z_audio, tile_hw_feature(z_fuse, self.shape)]}
