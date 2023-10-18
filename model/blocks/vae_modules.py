import torch
import torch.nn as nn
from utils.torch_utils import torch_L2normalize, torch_mean_kl_div, torch_reparametrize, tile_hw_feature, torch_tile


class encode_img_for_vae(nn.Module):
    def __init__(self, input_channels, config):
        super(encode_img_for_vae, self).__init__()
        self.cfg = config
        channels = self.cfg.MODEL.NECK_CHANNEL // 8
        latent_size = self.cfg.MODEL.VAE_LATENT_SIZE
        self.input_channels = input_channels
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(1*channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels*2)
        self.bn3 = nn.BatchNorm2d(channels*4)
        self.bn4 = nn.BatchNorm2d(channels*8)
        self.bn5 = nn.BatchNorm2d(channels*8)
        self.channel = channels
        self.hidden_shape = self.cfg.DATA.IMG_SIZE[0] // 32

        self.fc1 = nn.Linear(channels*8*self.hidden_shape*self.hidden_shape, latent_size)  # adjust according to input size
        self.fc2 = nn.Linear(channels*8*self.hidden_shape*self.hidden_shape, latent_size)  # adjust according to input size

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        output = self.leakyrelu(self.bn5(self.layer5(output)))
        output = output.view(-1, self.channel*8*self.hidden_shape*self.hidden_shape)  # adjust according to input size
        # output = self.tanh(output)

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return torch_reparametrize(mu, logvar), dist, [mu, logvar]
    

class encode_audio_for_vae(nn.Module):
    def __init__(self, config):
        super(encode_audio_for_vae, self).__init__()
        self.cfg = config
        channels = self.cfg.MODEL.NECK_CHANNEL
        latent_size = self.cfg.MODEL.VAE_LATENT_SIZE
        
        self.fc1 = nn.Linear(channels, latent_size)
        self.fc2 = nn.Linear(channels, latent_size)
        self.tanh = nn.Tanh()

    def forward(self, input):
        mu = self.fc1(input)
        logvar = self.fc2(input)
        dist = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return torch_reparametrize(mu, logvar), dist, [mu, logvar]
    
    
class encode_audio_with_x_for_vae(nn.Module):
    def __init__(self, input_channels, config):
        super(encode_audio_with_x_for_vae, self).__init__()
        self.cfg = config
        channels = self.cfg.MODEL.NECK_CHANNEL
        gt_encode_channels = self.cfg.MODEL.NECK_CHANNEL // 8
        latent_size = self.cfg.MODEL.VAE_LATENT_SIZE
        self.channel = gt_encode_channels
        self.hidden_shape = self.cfg.DATA.IMG_SIZE[0] // 32
        
        self.layer1 = nn.Conv2d(input_channels, gt_encode_channels, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(1*gt_encode_channels, 2*gt_encode_channels, kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(2*gt_encode_channels, 4*gt_encode_channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(4*gt_encode_channels, 4*gt_encode_channels, kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(4*gt_encode_channels, 4*gt_encode_channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(gt_encode_channels)
        self.bn2 = nn.BatchNorm2d(gt_encode_channels*2)
        self.bn3 = nn.BatchNorm2d(gt_encode_channels*4)
        self.bn4 = nn.BatchNorm2d(gt_encode_channels*4)
        self.bn5 = nn.BatchNorm2d(gt_encode_channels*4)
        self.leakyrelu = nn.LeakyReLU()
        
        self.fc1 = nn.Linear(channels+channels//4, latent_size)
        self.fc2 = nn.Linear(channels+channels//4, latent_size)
        self.fc_gt_vector = nn.Linear(self.channel*4*self.hidden_shape*self.hidden_shape, channels//4)
        self.tanh = nn.Tanh()

    def forward(self, input, gt):
        gt_enc = self.leakyrelu(self.bn1(self.layer1(gt)))
        gt_enc = self.leakyrelu(self.bn2(self.layer2(gt_enc)))
        gt_enc = self.leakyrelu(self.bn3(self.layer3(gt_enc)))
        gt_enc = self.leakyrelu(self.bn4(self.layer4(gt_enc)))
        gt_enc = self.leakyrelu(self.bn5(self.layer5(gt_enc)))
        gt_enc = gt_enc.view(-1, self.channel*4*self.hidden_shape*self.hidden_shape)
        gt_vector = self.fc_gt_vector(gt_enc)
        input_with_gt = torch.cat([input, gt_vector], dim=1)
        mu = self.fc1(input_with_gt)
        logvar = self.fc2(input_with_gt)
        dist = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return torch_reparametrize(mu, logvar), dist, [mu, logvar]
    
class noise_model(nn.Module):
    def __init__(self, cfg):
        super(noise_model, self).__init__()
        in_channel = cfg.MODEL.NECK_CHANNEL + cfg.MODEL.VAE_LATENT_SIZE + cfg.MODEL.VAE_LATENT_SIZE
        out_channel = cfg.MODEL.NECK_CHANNEL
        self.noise_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

    def process_z_noise(self, z, feat):
        spatial_axes = [2, 3]
        z_noise = torch.unsqueeze(z, 2)
        z_noise = torch_tile(z_noise, 2, feat.shape[spatial_axes[0]])
        z_noise = torch.unsqueeze(z_noise, 3)
        z_noise = torch_tile(z_noise, 3, feat.shape[spatial_axes[1]])

        return z_noise

    def forward(self, z1, z2, neck_features):
        z_noise_1 = self.process_z_noise(z1, neck_features[-1])
        z_noise_2 = self.process_z_noise(z2, neck_features[-1])
        neck_feat_with_noise = self.noise_conv(torch.cat((neck_features[-1], z_noise_1, z_noise_2), 1))
        neck_features[-1] = neck_feat_with_noise
        return neck_features
