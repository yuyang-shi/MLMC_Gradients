from __future__ import print_function
import torch
import torch.nn as nn

from .distributions import independent_normal
from torch.distributions import MultivariateNormal, kl_divergence


def compute_kl_multivariate_gaussian(mu_1, mu_2, cov_1, cov_2):
    return kl_divergence(MultivariateNormal(mu_1, cov_1), MultivariateNormal(mu_2, cov_2))


class LVM(nn.Module):
    def __init__(self, decoder, prior, encoder, px_z_dist_fn, pz_dist_fn, qz_x_dist_fn):
        super(LVM, self).__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.prior = prior

        self.px_z_dist = lambda z: px_z_dist_fn(*self.decoder(z))
        self.pz_dist = lambda: pz_dist_fn(*self.prior())
        self.qz_x_dist_fn = qz_x_dist_fn

        self.qz_x_dist = None
        self.qz_x_dist_detached = None

        # Used in trainer
        self.p_params = list(self.decoder.parameters()) + list(self.prior.parameters())
        self.q_params = list(self.encoder.parameters())

        self.register_buffer("_prior_fixed_samples", self.pz_dist().sample((64,)))

    def compute_qz_x_dist(self, *args):
        self.qz_x_dist = self.qz_x_dist_fn(*args)

    def compute_qz_x_dist_detached(self, *args):
        args_detach = []
        for arg in args:
            args_detach.append(arg.detach())
        self.qz_x_dist_detached = self.qz_x_dist_fn(*args_detach)

    def update_q(self, x):
        q_dist_args = self.encoder(x)
        self.compute_qz_x_dist(*q_dist_args)
        self.compute_qz_x_dist_detached(*q_dist_args)

    def sample_from_q(self, x, detach_q_samples=False):
        self.update_q(x)
        if detach_q_samples:
            z = self.qz_x_dist.sample()
            z.requires_grad_()
        else:
            z = self.qz_x_dist.rsample()
        return z

    def elbo(self, x, detach_q_params=False, detach_q_samples=False):
        z = self.sample_from_q(x, detach_q_samples)
        return self._elbo(x, z, detach_q_params)

    def _elbo(self, x, z, detach_q_params=False):
        if detach_q_params:
            log_qz_x = self.qz_x_dist_detached.log_prob(z)
        else:
            log_qz_x = self.qz_x_dist.log_prob(z)

        log_p_xz = self.pz_dist().log_prob(z) + self.px_z_dist(z).log_prob(x)

        return {
            "z": z,
            "log-p": log_p_xz,
            "log-q": log_qz_x,
            "elbo": log_p_xz - log_qz_x,
        }

    def fixed_sample(self):
        z = self._prior_fixed_samples
        return self.px_z_dist(z).sample()

    def compute_true_logprob(self, x):
        if type(self.prior).__name__ == "Constant_Normal_Constant_Sigma":
            prior_cov = (2 * self.prior.logsigma.expand(self.prior.net_dims[-1])).exp().diag()
            if type(self.decoder).__name__ in ("Linear_Normal_Constant_Sigma", "Linear_Normal_Constant_Diagonal"):
                if type(self.decoder).__name__ == "Linear_Normal_Constant_Sigma":
                    decoder_cov = (2*self.decoder.logsigma.expand(self.decoder.net_dims[-1])).exp().diag()
                else:
                    decoder_cov = (2*self.decoder.logdiagonal).exp().diag()

                px_dist = MultivariateNormal(self.decoder.linear_module_mu(self.prior.mu),
                                             decoder_cov + torch.chain_matmul(self.decoder.linear_module_mu.weight,
                                                                              prior_cov,
                                                                              self.decoder.linear_module_mu.weight.t()))
            # Passed test
            elif type(self.decoder).__name__ == "Identity_Normal_Constant_Sigma":
                decoder_cov = (2*self.decoder.logsigma.expand(self.decoder.net_dims[-1])).exp().diag()

                px_dist = MultivariateNormal(self.prior.mu, decoder_cov + prior_cov)

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        logprob = px_dist.log_prob(x).mean()
        return logprob

    def compute_true_kl_p_q(self, x):
        encoder_mu, encoder_sig = self.encoder(x)
        encoder_cov = torch.diag_embed(encoder_sig**2)
        if type(self.prior).__name__ == "Constant_Normal_Constant_Sigma":
            prior_cov = (2 * self.prior.logsigma.expand(self.prior.net_dims[-1])).exp().diag()
            if type(self.decoder).__name__ in ("Linear_Normal_Constant_Sigma", "Linear_Normal_Constant_Diagonal"):
                if type(self.decoder).__name__ == "Linear_Normal_Constant_Sigma":
                    decoder_cov = (2 * self.decoder.logsigma.expand(self.decoder.net_dims[-1])).exp().diag()
                else:
                    decoder_cov = (2 * self.decoder.logdiagonal).exp().diag()

                decoder_posterior_cov = (prior_cov.inverse() +
                                         torch.chain_matmul(self.decoder.linear_module_mu.weight.t(),
                                                            decoder_cov.inverse(),
                                                            self.decoder.linear_module_mu.weight)).inverse()
                decoder_posterior_mu = torch.matmul((prior_cov.inverse().matmul(self.prior.mu) +
                                                     (x - self.decoder.linear_module_mu.bias).matmul(
                                                         decoder_cov.inverse()).matmul(
                                                         self.decoder.linear_module_mu.weight)), decoder_posterior_cov)
            # Passed test
            elif type(self.decoder).__name__ == "Identity_Normal_Constant_Sigma":
                decoder_cov = (2 * self.decoder.logsigma.expand(self.decoder.net_dims[-1])).exp().diag()
                decoder_posterior_cov = (prior_cov.inverse() + decoder_cov.inverse()).inverse()
                decoder_posterior_mu = torch.matmul((prior_cov.inverse().matmul(self.prior.mu) +
                                                     x.matmul(decoder_cov.inverse())), decoder_posterior_cov)

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return compute_kl_multivariate_gaussian(decoder_posterior_mu, encoder_mu, decoder_posterior_cov, encoder_cov).mean()

    def get_p_grad(self):
        grads = []
        for param in self.p_params:
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        return grads

    def get_q_grad(self):
        grads = []
        for param in self.q_params:
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        return grads
