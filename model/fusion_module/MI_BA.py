import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks.base_blocks import BasicConv2d
from torch.distributions import Normal, Independent
from utils.torch_utils import torch_L2normalize, torch_kl_div, torch_reparametrize, tile_hw_feature


class MMILB(nn.Module):
    """Compute the Modality Mutual Information Lower Bound (MMILB) given bimodal representations.
    Args:
        x_size (int): embedding size of input modality representation x
        y_size (int): embedding size of input modality representation y
        mid_activation(int): the activation function in the middle layer of MLP
        last_activation(int): the activation function in the last layer of MLP that outputs logvar
    """
    def __init__(self, cfg, shape, mid_activation='ReLU', last_activation='Tanh'):
        super(MMILB, self).__init__()
        try:
            self.mid_activation = getattr(nn, mid_activation)
            self.last_activation = getattr(nn, last_activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
        self.input_channels = cfg.MODEL.NECK_CHANNEL
        self.hidden_channels = cfg.MODEL.MUTUAL_REG_CHANNEL
        self.latent_size = cfg.MODEL.MUTUAL_REG_SIZE
        self.rgb_channel_reduce = BasicConv2d(self.input_channels, self.hidden_channels, kernel_size=1)
        self.audio_channel_reduce = nn.Linear(self.input_channels, self.hidden_channels)
        self.shape = shape
        self.rgb_mu = nn.Sequential(
            nn.Linear(self.hidden_channels*shape[0]*shape[1], self.latent_size),
            self.mid_activation(),
            nn.Linear(self.latent_size, self.latent_size)
        )
        self.rgb_logvar = nn.Sequential(
            nn.Linear(self.hidden_channels*shape[0]*shape[1], self.latent_size),
            self.mid_activation(),
            nn.Linear(self.latent_size, self.latent_size)
        )
        self.audio_vector_mlp = nn.Sequential(
            nn.Linear(self.hidden_channels, self.latent_size),
            self.mid_activation(),
            nn.Linear(self.latent_size, self.latent_size)
        )

    def forward(self, rgb_feat, audio_feat):
        """ Forward lld (gaussian prior) and entropy estimation, partially refers the implementation
        of https://github.com/Linear95/CLUB/blob/master/MI_DA/MNISTModel_DANN.py
            Args:
                x (Tensor): x in above equation, shape (bs, x_size)
                y (Tensor): y in above equation, shape (bs, y_size)
        """
        BT, C, H, W = rgb_feat.shape
        audio_feat = audio_feat.permute(0, 2, 1).view(BT, C)
        rgb_feat = torch_L2normalize(rgb_feat, d=1)
        audio_feat = torch_L2normalize(audio_feat, d=1)
        rgb_feat = self.rgb_channel_reduce(rgb_feat)
        audio_feat = self.audio_channel_reduce(audio_feat)
        rgb_feat = rgb_feat.contiguous().view(-1, self.hidden_channels*H*W)
        audio_vector = torch.tanh(self.audio_vector_mlp(audio_feat))

        mu, logvar = torch.tanh(self.rgb_mu(rgb_feat)), torch.tanh(self.rgb_logvar(rgb_feat)) # (bs, hidden_size)

        positive = -0.5 * (mu - audio_vector)**2 / torch.exp(logvar)
        lld = torch.mean(torch.sum(positive, -1))

        return {"loss": lld}
