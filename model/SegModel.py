import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks.base_blocks import BasicConv2d
from model.blocks.rcab_block import RCAB
from model.fusion_module.tpavi import TPAVIModule
from model.fusion_module.avcorr import AVCorr
from model.fusion_module.MI_reg import Mutual_info_reg
from model.fusion_module.MI_BA import MMILB
from model.decoder.get_decoder import get_decoder
from model.encoder.get_encoder import get_encoder
from model.neck.get_neck import get_neck
from utils.torch_utils import torch_mean_kl_div, torch_reparametrize, torch_tile, compute_js_loss
from loss import DiffLoss
from model.blocks.vae_modules import encode_img_for_vae, encode_audio_for_vae, encode_audio_with_x_for_vae, noise_model


class CVAESeg(nn.Module):
    def __init__(self, config=None):
        super(CVAESeg, self).__init__()
        self.cfg = config
        self.neck_channel = self.cfg.MODEL.NECK_CHANNEL

        self.encoder, in_channel_list = get_encoder(self.cfg)
        self.neck_prior = get_neck(self.cfg, in_channel_list)
        self.neck_post = copy.deepcopy(self.neck_prior)
        self.decoder_prior = get_decoder(self.cfg)
        self.decoder_post = copy.deepcopy(self.decoder_prior)

        self.cvae_model = cvae_model(self.cfg)

    def forward(self, x, mask=None):
        backbone_features = self.encoder(x)
        neck_features_prior = self.neck_prior(backbone_features)
        neck_features_post = self.neck_post(backbone_features)
                    
        out_list = self.cvae_model(x, neck_features_prior, neck_features_post, mask)
        neck_features_sc_v_prior, neck_features_sc_v_post, loss_dict = out_list
        
        if mask is not None:   # In the training case with gt
            outputs_prior = self.decoder_prior(neck_features_sc_v_prior)
            outputs_post = self.decoder_post(neck_features_sc_v_post)

            return outputs_prior, outputs_post, loss_dict
        else:   # In the testing case without gt
            outputs_prior = self.decoder_prior(neck_features_sc_v_prior)
            return outputs_prior
        
    
class cvae_model(nn.Module):
    def __init__(self, config):
        super(cvae_model, self).__init__()
        self.cfg = config
        self.enc_vs = encode_img_for_vae(input_channels=3, config=config)
        self.enc_vs_y = encode_img_for_vae(input_channels=4, config=config)

        self.noise_model_prior = noise_model(config)
        self.noise_model_post = noise_model(config)

    def forward(self, img, neck_features_prior, neck_features_post, y=None):
        if y is None:
            ### encode the prior of image modality (s and c) (only input img)
            z_prior_vs, prior_vs, _ = self.enc_vs(img)
            
            neck_features_sc_v_prior = self.noise_model_prior(z_prior_vs, z_prior_vs, neck_features_prior)

            return neck_features_sc_v_prior, None, None
        else:
            # NOTE: How to implement for weak supervised manner
            ### encode the prior of image modality (s and c) (only input img)
            z_prior_vs, prior_vs, _ = self.enc_vs(img)  # BCHW

            ### encode the posteriors of image modality (s and c) (input img and gt)
            z_posterior_vs_y, posterior_vs_y, _ = self.enc_vs_y(torch.cat([img, y], 1))
                        
            # calculate latent loss
            latent_loss_v = torch_mean_kl_div(posterior_vs_y, prior_vs)
            kl_loss = latent_loss_v

            neck_features_sc_v_prior = self.noise_model_prior(z_prior_vs, z_prior_vs, neck_features_prior)
            neck_features_sc_v_post = self.noise_model_post(z_posterior_vs_y, z_posterior_vs_y, neck_features_post)

            return neck_features_sc_v_prior, neck_features_sc_v_post, {"kl_loss": kl_loss}



def prior_expert(size):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = torch.zeros(size)
    logvar = torch.log(torch.ones(size))

    mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        T = 1 / (var + eps)  # precision of i-th Gaussian expert at point x
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


class mm_vae_with_poe(nn.Module):
    def __init__(self, config):
        super(mm_vae_with_poe, self).__init__()
        self.cfg = config
        self.enc_vs = encode_img_for_vae(input_channels=3, config=config)
        self.enc_as = encode_audio_for_vae(config=config)
        self.enc_vac = encode_audio_with_x_for_vae(input_channels=3, config=config)
        
        self.enc_vs_y = encode_img_for_vae(input_channels=4, config=config)
        self.enc_as_y = encode_audio_with_x_for_vae(input_channels=1, config=config)
        self.enc_vac_y = encode_audio_with_x_for_vae(input_channels=4, config=config)
        
        # self.sc1_prior = nn.Linear(self.cfg.MODEL.VAE_LATENT_SIZE*2, self.cfg.MODEL.VAE_LATENT_SIZE)
        # self.sc2_prior = nn.Linear(self.cfg.MODEL.VAE_LATENT_SIZE*2, self.cfg.MODEL.VAE_LATENT_SIZE)
        
        # self.sc1_posterior = nn.Linear(self.cfg.MODEL.VAE_LATENT_SIZE*2, self.cfg.MODEL.VAE_LATENT_SIZE)
        # self.sc2_posterior = nn.Linear(self.cfg.MODEL.VAE_LATENT_SIZE*2, self.cfg.MODEL.VAE_LATENT_SIZE)

        self.noise_model_prior = noise_model(config)
        self.noise_model_post = noise_model(config)
        
        self.ba_lower_bound_mu_post = nn.Linear(self.cfg.MODEL.VAE_LATENT_SIZE, self.cfg.MODEL.MUTUAL_REG_SIZE)
        self.ba_lower_bound_logvar_post = nn.Linear(self.cfg.MODEL.VAE_LATENT_SIZE, self.cfg.MODEL.MUTUAL_REG_SIZE)
        self.ba_lower_bound_mu_prior = nn.Linear(self.cfg.MODEL.VAE_LATENT_SIZE, self.cfg.MODEL.MUTUAL_REG_SIZE)
        self.ba_lower_bound_logvar_prior = nn.Linear(self.cfg.MODEL.VAE_LATENT_SIZE, self.cfg.MODEL.MUTUAL_REG_SIZE)
        
        self.ba_lower_bound_sc_a_post = nn.Linear(self.cfg.MODEL.VAE_LATENT_SIZE, self.cfg.MODEL.MUTUAL_REG_SIZE)
        self.ba_lower_bound_sc_a_prior = nn.Linear(self.cfg.MODEL.VAE_LATENT_SIZE, self.cfg.MODEL.MUTUAL_REG_SIZE)
        self.diff_loss = DiffLoss()
        self.experts = ProductOfExperts()

    def forward(self, img, audio, neck_features_prior, neck_features_post, y=None):
        if y is None:
            ### encode the prior of image modality (s and c) (only input img)
            z_prior_vs, prior_vs, prior_mu_var_vs = self.enc_vs(img)
            ### encode the prior of audio modality (s and c) (only input audio)
            z_prior_as, prior_as, prior_mu_var_as = self.enc_as(audio)
            ### !![need to encode a v_a_c, not v_c and a_c]
            z_prior_vac, prior_vac, prior_mu_var_vac = self.enc_vac(audio, img)
            
            # sc_v_prior = self.sc1_prior(torch.cat([z_prior_vs, z_prior_vac], dim=1))
            # sc_a_prior = self.sc2_prior(torch.cat([z_prior_as, z_prior_vac], dim=1))   # use MI loss
            mu_sc1, logvar_sc1 = prior_expert((1, img.shape[0], self.cfg.MODEL.VAE_LATENT_SIZE))
            mu_sc2, logvar_sc2 = prior_expert((1, img.shape[0], self.cfg.MODEL.VAE_LATENT_SIZE))

            sc1_v_mu = torch.cat([mu_sc1, prior_mu_var_vs[0].unsqueeze(0), \
                prior_mu_var_vac[0].unsqueeze(0)], dim=0)
            sc1_v_var = torch.cat([logvar_sc1, prior_mu_var_vs[1].unsqueeze(0), \
                prior_mu_var_vac[1].unsqueeze(0)], dim=0)
            sc1_v_mu_poe, sc1_v_var_poe = self.experts(sc1_v_mu, sc1_v_var)
            sc_v_prior = torch_reparametrize(sc1_v_mu_poe, sc1_v_var_poe)

            sc2_v_mu = torch.cat([mu_sc2, prior_mu_var_as[0].unsqueeze(0), \
                prior_mu_var_vac[0].unsqueeze(0)], dim=0)
            sc2_v_var = torch.cat([logvar_sc2, prior_mu_var_as[1].unsqueeze(0), \
                prior_mu_var_vac[1].unsqueeze(0)], dim=0)
            sc2_v_mu_poe, sc2_v_var_poe = self.experts(sc2_v_mu, sc2_v_var)
            sc_a_prior = torch_reparametrize(sc2_v_mu_poe, sc2_v_var_poe)
            
            neck_features_sc_v_prior = self.noise_model_prior(sc_v_prior, sc_a_prior, neck_features_prior)

            return neck_features_sc_v_prior, None, None
        else:
            # NOTE: How to implement for weak supervised manner
            ### encode the prior of image modality (s and c) (only input img)
            z_prior_vs, prior_vs, prior_mu_var_vs = self.enc_vs(img)  # BCHW
            ### encode the prior of audio modality (s and c) (only input audio)
            z_prior_as, prior_as, prior_mu_var_as = self.enc_as(audio)  # BC
            ### !![need to encode a v_a_c, not v_c and a_c]
            z_prior_vac, prior_vac, prior_mu_var_vac = self.enc_vac(audio, img)

            ### encode the posteriors of image modality (s and c) (input img and gt)
            z_posterior_vs_y, posterior_vs_y, post_mu_var_vs = self.enc_vs_y(torch.cat([img, y], 1))
            ### encode the posteriors of audio modality (s and c) (input audio and gt)
            z_posterior_as_y, posterior_as_y, post_mu_var_as = self.enc_as_y(audio, y)
            ### !![need to encode a v_a_c, not v_c and a_c]
            z_posterior_vac_y, posterior_vac_y, post_mu_var_vac = self.enc_vac_y(audio, torch.cat([img, y], 1))
            # calculate latent loss
            latent_loss_v = torch_mean_kl_div(posterior_vs_y, prior_vs)
            latent_loss_a = torch_mean_kl_div(posterior_as_y, prior_as)
            latent_loss_va = compute_js_loss(prior_mu_var_vac[0], prior_mu_var_vac[1], 
                                             post_mu_var_vac[0], post_mu_var_vac[1])   # use JS loss?
            # import pdb; pdb.set_trace()
            kl_loss = latent_loss_v + latent_loss_a + latent_loss_va

            mu_sc1, logvar_sc1 = prior_expert((1, img.shape[0], self.cfg.MODEL.VAE_LATENT_SIZE))
            mu_sc2, logvar_sc2 = prior_expert((1, img.shape[0], self.cfg.MODEL.VAE_LATENT_SIZE))

            sc1_v_mu = torch.cat([mu_sc1, prior_mu_var_vs[0].unsqueeze(0), \
                prior_mu_var_vac[0].unsqueeze(0)], dim=0)
            sc1_v_var = torch.cat([logvar_sc1, prior_mu_var_vs[1].unsqueeze(0), \
                prior_mu_var_vac[1].unsqueeze(0)], dim=0)
            sc1_v_mu_poe, sc1_v_var_poe = self.experts(sc1_v_mu, sc1_v_var)
            sc_v_prior = torch_reparametrize(sc1_v_mu_poe, sc1_v_var_poe)

            sc2_v_mu = torch.cat([mu_sc2, prior_mu_var_as[0].unsqueeze(0), \
                prior_mu_var_vac[0].unsqueeze(0)], dim=0)
            sc2_v_var = torch.cat([logvar_sc2, prior_mu_var_as[1].unsqueeze(0), \
                prior_mu_var_vac[1].unsqueeze(0)], dim=0)
            sc2_v_mu_poe, sc2_v_var_poe = self.experts(sc2_v_mu, sc2_v_var)
            sc_a_prior = torch_reparametrize(sc2_v_mu_poe, sc2_v_var_poe)
            ###
            sc1_v_mu = torch.cat([mu_sc1, post_mu_var_vs[0].unsqueeze(0), \
                post_mu_var_vac[0].unsqueeze(0)], dim=0)
            sc1_v_var = torch.cat([logvar_sc1, post_mu_var_vs[1].unsqueeze(0), \
                post_mu_var_vac[1].unsqueeze(0)], dim=0)
            sc1_v_mu_poe, sc1_v_var_poe = self.experts(sc1_v_mu, sc1_v_var)
            sc_v_posterior = torch_reparametrize(sc1_v_mu_poe, sc1_v_var_poe)

            sc2_v_mu = torch.cat([mu_sc2, post_mu_var_as[0].unsqueeze(0), \
                post_mu_var_vac[0].unsqueeze(0)], dim=0)
            sc2_v_var = torch.cat([logvar_sc2, post_mu_var_as[1].unsqueeze(0), \
                post_mu_var_vac[1].unsqueeze(0)], dim=0)
            sc2_v_mu_poe, sc2_v_var_poe = self.experts(sc2_v_mu, sc2_v_var)
            sc_a_posterior = torch_reparametrize(sc2_v_mu_poe, sc2_v_var_poe)

            # sc_v_prior = self.sc1_prior(torch.cat([z_prior_vs, z_prior_vac], dim=1))
            # sc_a_prior = self.sc2_prior(torch.cat([z_prior_as, z_prior_vac], dim=1))   # use MI loss
            
            # sc_v_posterior = self.sc1_posterior(torch.cat([z_posterior_vs_y, z_posterior_vac_y], dim=1))
            # sc_a_posterior = self.sc2_posterior(torch.cat([z_posterior_as_y, z_posterior_vac_y], dim=1))   # use MI loss
            
            sc_v_posterior_mu = torch.tanh(self.ba_lower_bound_mu_post(sc_v_posterior))
            sc_v_posterior_logvar = torch.tanh(self.ba_lower_bound_logvar_post(sc_v_posterior))
            sc_v_prior_mu = torch.tanh(self.ba_lower_bound_mu_prior(sc_v_prior))
            sc_v_prior_logvar = torch.tanh(self.ba_lower_bound_logvar_prior(sc_v_prior))
            sc_a_posterior_mi = torch.tanh(self.ba_lower_bound_sc_a_post(sc_a_posterior))
            sc_a_prior_mi = torch.tanh(self.ba_lower_bound_sc_a_prior(sc_a_prior))
            
            positive_posterior = -0.5 * (sc_v_posterior_mu - sc_a_posterior_mi)**2 / torch.exp(sc_v_posterior_logvar)
            positive_prior = -0.5 * (sc_v_prior_mu - sc_a_prior_mi)**2 / torch.exp(sc_v_prior_logvar)
            lld = torch.mean(torch.sum(positive_posterior, -1)) + torch.mean(torch.sum(positive_prior, -1))
            diff_prior = self.diff_loss(z_prior_vs, z_prior_vac) + self.diff_loss(z_prior_as, z_prior_vac)
            diff_post = self.diff_loss(z_posterior_vs_y, z_posterior_vac_y) + self.diff_loss(z_posterior_as_y, z_posterior_vac_y)
            diff_loss = diff_prior + diff_post
            # import pdb; pdb.set_trace()

            neck_features_sc_v_prior = self.noise_model_prior(sc_v_prior, sc_a_prior, neck_features_prior)
            neck_features_sc_v_post = self.noise_model_post(sc_v_posterior, sc_a_posterior, neck_features_post)

            return neck_features_sc_v_prior, neck_features_sc_v_post, \
                {"kl_loss": kl_loss, "mi_loss": lld, "diff_loss": diff_loss}