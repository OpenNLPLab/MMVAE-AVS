import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks.base_blocks import BasicConv2d
from model.blocks.rcab_block import RCAB
from model.blocks.neck_blocks import ASPP_Module


class rcab_decoder(torch.nn.Module):
    def __init__(self, neck_channel):
        super(rcab_decoder, self).__init__()
        self.channel_size = neck_channel
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_reformat_2 = BasicConv2d(in_planes=self.channel_size, out_planes=self.channel_size, kernel_size=1)
        self.conv_reformat_3 = BasicConv2d(in_planes=self.channel_size*2, out_planes=self.channel_size, kernel_size=1)
        self.conv_reformat_4 = BasicConv2d(in_planes=self.channel_size*2, out_planes=self.channel_size, kernel_size=1)
        
        self.racb_4, self.racb_3 = RCAB(self.channel_size*2), RCAB(self.channel_size*2)
        self.racb_2, self.racb_1 = RCAB(self.channel_size*2), RCAB(self.channel_size*1)

        self.layer5 = self._make_pred_layer(ASPP_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, self.channel_size*2)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, features):
        x1, x2, x3, x4 = features[0], features[1], features[2], features[3]

        feat_cat_1 = self.racb_1(x4)
        feat2 = self.conv_reformat_2(feat_cat_1)

        feat_cat_2 = torch.cat((x3, self.upsample2(feat2)), 1)
        feat_cat_2 = self.racb_2(feat_cat_2)
        feat3 = self.conv_reformat_3(feat_cat_2)

        feat_cat_3 = torch.cat((x2, self.upsample2(feat3)), 1)
        feat_cat_3 = self.racb_3(feat_cat_3)
        feat4 = self.conv_reformat_4(feat_cat_3)

        feat_cat_4 = torch.cat((x1, self.upsample2(feat4)), 1)
        feat_cat_4 = self.racb_4(feat_cat_4)

        output4 = F.upsample(self.layer5(feat_cat_4), scale_factor=4, mode='bilinear', align_corners=True)

        return output4
