import torch
from torch import nn
from torch.nn import functional as F
from utils.torch_utils import torch_L2normalize


class TPAVIModule(nn.Module):
    def __init__(self, in_channels):
        super(TPAVIModule, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        # function g in the paper which goes through conv. with kernel size 1
        self.g = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # define theta and phi for all operations except gaussian
        self.theta = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.norm_layer = nn.LayerNorm(in_channels)
        
        self.W_z = nn.Sequential(
                nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                nn.BatchNorm3d(self.in_channels)
            )
        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

    def forward(self, visual, audio):
        """
            x: [B*T, C, H, W] first transfor to [B, C, T, H, W], T=5
            audio: [B*T, C]
        """
        BT, C, H, W = visual.shape
        B, T = BT//5, 5
        visual = visual.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
        audio = audio.reshape(B, T, C).permute(0, 2, 1) # [B, C, T]

        audio_temp, visual = torch_L2normalize(audio, d=1), torch_L2normalize(visual, d=1)
        audio = audio_temp[..., None, None].repeat(1, 1, 1, H, W)  # [bs, C, T] -> [bs, C, T, 1, 1] -> [bs, C, T, H, W]

        g_x = self.g(visual).view(B, self.inter_channels, -1).permute(0, 2, 1) # [bs, C, THW] -> [bs, THW, C]

        theta_x = self.theta(visual).view(B, self.inter_channels, -1) # [bs, C', THW]
        phi_x = self.phi(audio).view(B, self.inter_channels, -1) # [bs, C', THW]
        f = torch.matmul(theta_x.permute(0, 2, 1), phi_x) # [bs, THW, THW]

        f_div_C = f / f.size(-1)  # [bs, THW, THW]
        y = torch.matmul(f_div_C, g_x) # [bs, THW, C]
        y = y.permute(0, 2, 1).contiguous().view(B, self.inter_channels, T, H, W)  # [bs, C, THW] -> [bs, C', T, H, W]
        
        W_y = self.W_z(y)  # [bs, C, T, H, W]
        z = W_y + visual #  # [bs, C, T, H, W]

        # add LayerNorm
        z = z.permute(0, 2, 3, 4, 1) # [bs, T, H, W, C]
        z = self.norm_layer(z)
        z = z.permute(0, 1, 4, 2, 3).view(-1, C, H, W) # [bs, T, H, W, C] -> [bs, T, C, H, W] -> [bs*T, C, H, W]

        return z, audio_temp
