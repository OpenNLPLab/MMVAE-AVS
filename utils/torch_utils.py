import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl


def torch_L2normalize(x, d=1):
    eps = 1e-6
    norm = x ** 2
    norm = norm.sum(dim=d, keepdim=True) + eps
    norm = norm ** (0.5)
    return (x / norm)


def torch_kl_div(posterior_latent_space, prior_latent_space):
    kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
    return kl_div


def torch_mean_kl_div(posterior_latent_space, prior_latent_space):
    kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
    return torch.mean(kl_div)


def torch_mean_js_div(posterior_latent_space, prior_latent_space):
    m = 0.5 * (posterior_latent_space + prior_latent_space)
    js_div = torch_mean_kl_div(posterior_latent_space, m) + torch_mean_kl_div(prior_latent_space, m)
    return js_div


def compute_js_loss(source_mu, source_log_var, target_mu, target_log_var):
    def get_prob(mu, log_var):
        dist = Normal(mu, torch.exp(0.5 * log_var))
        val = dist.sample()
        return dist.log_prob(val).exp()

    def kl_loss(p, q):
        return F.kl_div(p, q, reduction="batchmean", log_target=False)

    source_prob = get_prob(source_mu, source_log_var)
    target_prob = get_prob(target_mu, target_log_var)

    log_mean_prob = (0.5 * (source_prob + target_prob)).log()
    js_loss = 0.5 * (kl_loss(log_mean_prob, source_prob) + kl_loss(log_mean_prob, target_prob))

    return js_loss


def torch_tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    index_list = [init_dim * np.arange(n_tile) + i for i in range(init_dim)]
    order_index = torch.LongTensor(np.concatenate(index_list)).to(a.device)
    return torch.index_select(a, dim, order_index)


def torch_reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    eps = Variable(eps)
    return eps.mul(std).add_(mu)


def tile_hw_feature(feat, shape):
    feat = torch_tile(feat.unsqueeze(-1), 2, shape[0])
    feat = torch_tile(feat.unsqueeze(-1), 3, shape[1])
    
    return feat
