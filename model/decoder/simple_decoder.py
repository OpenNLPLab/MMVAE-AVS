import torch
import torch.nn as nn
from model.blocks.base_blocks import FeatureFusionBlock, SimpleHead


class simple_decoder(torch.nn.Module):
    def __init__(self, neck_channel):
        super(simple_decoder, self).__init__()
        self.channel_size = neck_channel

        self.short_connect_with_decode_module = nn.ModuleList()
        for _ in range(4):
            self.short_connect_with_decode_module.append(FeatureFusionBlock(self.channel_size))

        self.head_up_2 = SimpleHead(channel=self.channel_size, rate=2)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, features):
        features = features[::-1]

        conv_feat_list = []
        for i in range(len(features)):
            if i == 0:
                conv_feat = self.short_connect_with_decode_module[i](features[i])
            else:
                conv_feat = self.short_connect_with_decode_module[i](conv_feat, features[i])
            conv_feat_list.append(conv_feat)

        output4 = self.head_up_2(conv_feat_list[3])        

        return output4


class simple_decoder_no_modulelist(torch.nn.Module):
    def __init__(self, neck_channel):
        super(simple_decoder_no_modulelist, self).__init__()
        self.channel_size = neck_channel

        self.fusion_block_1 = FeatureFusionBlock(self.channel_size)
        self.fusion_block_2 = FeatureFusionBlock(self.channel_size)
        self.fusion_block_3 = FeatureFusionBlock(self.channel_size)
        self.fusion_block_4 = FeatureFusionBlock(self.channel_size)

        self.head_up_2 = SimpleHead(channel=self.channel_size, rate=2)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, features):
        features = features[::-1]

        conv_feat = self.fusion_block_1(features[0])
        conv_feat = self.fusion_block_2(conv_feat, features[1])
        conv_feat = self.fusion_block_3(conv_feat, features[2])
        conv_feat = self.fusion_block_4(conv_feat, features[3])

        output4 = self.head_up_2(conv_feat)        

        return output4