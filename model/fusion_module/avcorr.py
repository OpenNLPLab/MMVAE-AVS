import torch
from torch import nn
from utils.torch_utils import torch_L2normalize
from model.blocks.base_blocks import BasicConv2d


class AVCorr(nn.Module):
    def __init__(self, in_channels):
        super(AVCorr, self).__init__()

        self.in_channels = in_channels
        self.conv1 = BasicConv2d(self.in_channels, self.in_channels//4, 1, norm=False)
        self.conv2 = BasicConv2d(self.in_channels//4, self.in_channels//4, 3, padding=1, norm=False)
        self.conv3 = BasicConv2d(self.in_channels//4, self.in_channels, 1, norm=False)

    def forward(self, visual, audio):
        """
            x: [B*T, C, H, W] first transfor to [B, C, T, H, W], T=5
            audio: [B*T, C]
        """
        BT, C, H, W = visual.shape
        B, T = BT//5, 5

        audio_temp, visual = torch_L2normalize(audio, d=1), torch_L2normalize(visual, d=1)
        corr = torch.einsum('ncqa,nchw->ncqa', [visual, audio_temp.unsqueeze(2).unsqueeze(3)])
        corr_norm = torch.sigmoid(self.conv3(self.conv2(self.conv1(corr))))
        visual_atten = visual * corr_norm
        visual_out = visual + visual_atten

        return visual_out, audio_temp.reshape(B, T, C).permute(0, 2, 1)
