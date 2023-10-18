import torch.nn as nn


class UpsampleDeterministic(nn.Module):
    def __init__(self, upscale=2):
        super(UpsampleDeterministic, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        '''
        x: 4-dim tensor. shape is (batch,channel,h,w)
        output: 4-dim tensor. shape is (batch,channel,self.upscale*h,self.upscale*w)
        '''
        return x[:, :, :, None, :, None].expand(-1, -1, -1, self.upscale, -1, self.upscale).reshape(x.size(0), x.size(1), x.size(2)*self.upscale, x.size(3)*self.upscale)
    
    
def upsample_deterministic(x, upscale):
    return x[:, :, :, None, :, None].expand(-1, -1, -1, upscale, -1, upscale).reshape(x.size(0), x.size(1), x.size(2)*upscale, x.size(3)*upscale)
