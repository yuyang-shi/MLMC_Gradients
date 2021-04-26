import numpy as np
import torch
from torch.distributions import Categorical, Geometric
from functools import partial
from arviz.stats import psislw
from tqdm import tqdm


def metrics(density, x, num_elbo_samples, verbose=False, train_metrics=None, decay_rate_test_num_samples=None,
            decay_rate_test_K_max=None, decay_rate_test_grad_idx=None,
            tvo_integrand_shape_test_num_partitions=None, if_sumo=False, if_tvo=False):
    with torch.no_grad():
        log_p, log_q = compute_log_p_q(density, x, max(64, num_elbo_samples), True, True)
        log_w = log_p - log_q

        elbo = log_w.mean(dim=1)

        iwae64 = logmeanexp(log_w[:, :64], dim=1)
        log_prob = logmeanexp(log_w[:, :num_elbo_samples], dim=1)

        dim = int(np.prod(x.shape[1:]))
        bpd = -log_prob / dim / np.log(2)

        elbo_gap = log_prob - elbo

        forward_kl = torch.sum(lognormexp(log_w, dim=1).exp() * log_w, dim=1) - log_prob

        chi_squared = (((log_w - log_prob.unsqueeze(1)).exp() - 1) ** 2).mean(dim=1) ** (1 / 2)

        metrics_results = {
            "elbo": elbo,
            "log-prob": log_prob,
            "iwae64": iwae64,
            "iwae" + str(num_elbo_samples): log_prob,
            "iwae-" + str(num_elbo_samples) + "-64-gap": log_prob - iwae64,
            "bpd": bpd,
            "elbo-gap": elbo_gap,
            "forward-kl": forward_kl,
            "chi-squared": chi_squared
        }

        if train_metrics is not None:
            log_w, _, _ = train_metrics.get_log_w(density, x, 500, tempered=True)

            _, khats = psislw(log_w.clone().detach().cpu().numpy()[:, :, 0])
            metrics_results["khat"] = torch.from_numpy(khats).to(log_w.device).view(-1, 1)
        if not verbose:
            return metrics_results

        train_metrics_p = - train_metrics(density, x)["losses"]["p"]

        # Tempered log_w
        log_w, _, _ = train_metrics.get_log_w(density, x, 5000, tempered=True)

        ess = torch.exp(2 * (log_w.logsumexp(dim=1)) - (log_w * 2).logsumexp(dim=1))

        _, khats = psislw(log_w.clone().detach().cpu().numpy()[:, :, 0])

    nll_delta_squared_dict, nll_delta_batch_avg_squared_dict, p_grad_delta_batch_avg_squared_dict = delta_decay_rate_test(
        density, x, train_metrics, decay_rate_test_num_samples, decay_rate_test_K_max, decay_rate_test_grad_idx,
        if_sumo)

    tvo_integrand_dict = {}
    if if_tvo:
        tvo_integrand_dict = tvo_integrand_shape_test(density, x, train_metrics,
                                                      tvo_integrand_shape_test_num_partitions)

    return {
        **metrics_results,
        "train-metrics-p": train_metrics_p,
        "ess": ess,
        "khat": torch.from_numpy(khats).to(log_w.device).view(-1, 1),
        **nll_delta_squared_dict,
        **nll_delta_batch_avg_squared_dict,
        **p_grad_delta_batch_avg_squared_dict,
        **tvo_integrand_dict
    }


def delta_decay_rate_test(density, x, train_metrics, num_samples, K_max, test_grad_idx, if_sumo=False):
    nll_delta_squared_dict = {}
    nll_delta_squared_batch_avg_dict = {}
    p_grad_delta_squared_batch_avg_dict = {}

    nll_delta_ks = torch.empty((x.shape[0], num_samples, K_max + 1), device=x.device)
    p_grad_delta_ks = torch.empty((num_samples, K_max + 1), device=x.device)

    for i in range(num_samples):
        log_w, log_p, log_q = train_metrics.get_log_w(density, x, 2 ** (K_max + 1))
        for k in range(0, K_max + 1):
            log_w_reshape = log_w[:, :2 ** (k + 1)]
            log_p_reshape = log_p[:, :2 ** (k + 1)]
            log_q_reshape = log_q[:, :2 ** (k + 1)]
            upper_level_term = train_metrics.p_train_function(
                log_w=log_w_reshape, log_p=log_p_reshape, log_q=log_q_reshape)

            if if_sumo:
                log_w_reshape = log_w_reshape[:, :2 ** k]
                lower_level_term = train_metrics.p_train_function(log_w=log_w_reshape)
            else:
                log_w_reshape = log_w_reshape.view(log_w_reshape.shape[0], 2, 2 ** k, 1)
                log_p_reshape = log_p_reshape.view(log_w_reshape.shape[0], 2, 2 ** k, 1)
                log_q_reshape = log_q_reshape.view(log_w_reshape.shape[0], 2, 2 ** k, 1)
                lower_level_term = train_metrics.p_train_function(
                    log_w=log_w_reshape, log_p=log_p_reshape, log_q=log_q_reshape).mean(dim=1)  # shape (x.shape[0], 1)

            delta_k = upper_level_term - lower_level_term  # shape (x.shape[0], 1)

            nll_delta_ks[:, i, k] = delta_k[:, 0].detach()

            delta_k_mean = delta_k.mean()

            delta_k_mean.backward(retain_graph=True)

            p_grad_delta_ks[i, k] = density.get_p_grad()[test_grad_idx].detach()

            density.zero_grad()

    for k in range(0, K_max + 1):
        nll_delta_squared_dict["nll-Delta-" + str(k) + "-squared"] = \
            nll_delta_ks[:, :, k].square().mean(dim=1, keepdim=True)  # shape (x.shape[0], 1)
        nll_delta_squared_batch_avg_dict["nll-Delta-" + str(k) + "-squared-batch-averaged"] = \
            nll_delta_ks[:, :, k].mean(dim=0, keepdim=True).square().mean(1, keepdim=True)  # shape (1, 1)
        p_grad_delta_squared_batch_avg_dict["p-grad-Delta-" + str(k) + "-squared-batch-averaged"] = \
            p_grad_delta_ks[:, k].square().mean(dim=0, keepdim=True).unsqueeze(0)  # shape (1, 1)

    return nll_delta_squared_dict, nll_delta_squared_batch_avg_dict, p_grad_delta_squared_batch_avg_dict


def tvo_integrand_shape_test(density, x, train_metrics, num_partitions):
    tvo_integrand_dict = {}

    partition = _get_partition(num_partitions, "linear", None).to(x.device)

    with torch.no_grad():
        log_p, log_q = compute_log_p_q(density, x, 5000, True, True)
        log_w = log_p - log_q

    heated_log_w = log_w * partition  # shape (x.shape[0], 5000, num_partitions)
    heated_normalized_w = lognormexp(heated_log_w, dim=-2).exp()

    integrands = torch.sum(heated_normalized_w * log_w, dim=-2)

    for k in range(num_partitions + 1):
        tvo_integrand_dict["tvo-integrand-beta-" + str(k)] = integrands[:, k]

    for k in [5, 10, 20, 50]:
        iwae_k = logmeanexp(log_w[:, :k], dim=1)
        tvo_integrand_dict["iwae" + str(k)] = iwae_k

    return tvo_integrand_dict


def logmeanexp(values, dim, keepdim=False):
    return torch.logsumexp(values, dim, keepdim) - np.log(values.shape[dim])


def lognormexp(values, dim):
    log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
    return values - log_denominator


def compute_log_p_q(density, x, num_importance_samples, detach_q_params, detach_q_samples, z=None, return_z=False):
    if z is None:
        x_samples = x.repeat_interleave(num_importance_samples, dim=0).view(x.shape[0], num_importance_samples,
                                                                            *x.shape[1:])
        result = density.elbo(
            x_samples,
            detach_q_params=detach_q_params,
            detach_q_samples=detach_q_samples
        )
        output_shape = (x.shape[0], num_importance_samples, 1)

    else:
        num_importance_samples = np.prod(z.shape[1:-1])
        x_samples = x.repeat_interleave(num_importance_samples, dim=0).view(x.shape[0], *z.shape[1:-1], *x.shape[1:])
        result = density._elbo(x_samples, z, detach_q_params=detach_q_params)
        output_shape = [x.shape[0]] + list(z.shape[1:-1]) + [1]

    log_p = result["log-p"].view(output_shape)
    log_q = result["log-q"].view(output_shape)

    if not return_z:
        return log_p, log_q

    else:
        z = result["z"]
        return log_p, log_q, z


def get_detach_config(q_train_method):
    if q_train_method in ("elbo", "iwae", "var"):
        config = {"detach_q_params": False, "detach_q_samples": False}
    elif q_train_method in ("rws", "unbiased rws", "tvo"):
        config = {"detach_q_params": False, "detach_q_samples": True}
    elif q_train_method in ("rws_stl", "unbiased rws_stl"):
        config = {"detach_q_params": True, "detach_q_samples": False}
    else:
        assert False, f"Invalid q train method `{q_train_method}'"
    return config


def _iwae(log_w, **kwargs):
    return - logmeanexp(log_w, dim=-2)


def _elbo(log_w, **kwargs):
    return -log_w.mean(dim=-2)


class ELBO:
    def __init__(self, num_importance_samples, q_train_method):
        self.num_importance_samples = num_importance_samples
        self.q_train_method = q_train_method
        self.p_train_function = _elbo

    def __call__(self, density, x, update_K=True):
        self.get_log_w(density, x, self.num_importance_samples)
        p_loss = self.p_train_function(self.log_w).mean()

        self.get_log_w(density, x, self.num_importance_samples)
        if self.q_train_method == "elbo":
            q_loss = self.p_train_function(self.log_w).mean()
        else:
            assert False, f"Invalid q train method `{self.q_train_method}'"

        return {"losses": {"p": p_loss, "q": q_loss}}

    def _update_K(self):
        pass

    def get_log_w(self, density, x, num_importance_samples, **kwargs):
        self.log_p, self.log_q = compute_log_p_q(density, x, num_importance_samples,
                                                 **get_detach_config(self.q_train_method))
        self.log_w = self.log_p - self.log_q
        return self.log_w, self.log_p, self.log_q


class IWAE:
    def __init__(self, num_importance_samples, q_train_method):
        self.num_importance_samples = num_importance_samples
        self.q_train_method = q_train_method
        self.p_train_function = _iwae

    def __call__(self, density, x, update_K=True):
        self.get_log_w(density, x, self.num_importance_samples)
        p_loss = self.p_train_function(self.log_w).mean()

        self.get_log_w(density, x, self.num_importance_samples)
        if self.q_train_method in ("iwae", "rws", "rws_stl"):
            q_loss = _compute_q_loss(self.log_w, self.q_train_method).mean()
        else:
            assert False, f"Invalid q train method `{self.q_train_method}'"

        return {"losses": {"p": p_loss, "q": q_loss}}

    def _update_K(self):
        pass

    def get_log_w(self, density, x, num_importance_samples, **kwargs):
        self.log_p, self.log_q = compute_log_p_q(density, x, num_importance_samples,
                                                 **get_detach_config(self.q_train_method))
        self.log_w = self.log_p - self.log_q
        return self.log_w, self.log_p, self.log_q


def _compute_q_loss(log_w, q_train_method):
    """
    Reduces the -2nd dimension of log_w and computes the q loss function of q_train_method
    """
    if q_train_method == "rws":
        q_loss = logmeanexp(log_w, dim=-2)
    elif q_train_method == "iwae":
        q_loss = - logmeanexp(log_w, dim=-2)
    elif q_train_method == "elbo":
        q_loss = - log_w.mean(dim=-2)
    elif q_train_method == "rws_stl":
        normalized_weights = lognormexp(log_w, dim=-2).exp()
        q_loss = torch.sum(- normalized_weights.detach() * log_w, dim=-2)
    else:
        assert False, f"Invalid q train method `{q_train_method}'"
    return q_loss


# Class for both SUMO and ML-LL
class UnbiasedEstimator:
    def __init__(self, geom_prob, K_cap, I0_start_level, q_train_method, q_num_importance_samples):
        self.geom_prob = geom_prob
        self.K_cap = K_cap
        self.I0_start_level = I0_start_level
        self.q_train_method = q_train_method
        self.q_num_importance_samples = q_num_importance_samples
        self.p_train_function = _iwae

        if self.K_cap == np.inf:
            self.K_dist = Geometric(probs=geom_prob)
            self.K_pmf = lambda k: self.geom_prob * (1 - self.geom_prob) ** k
            self.K_reverse_cdf = lambda k: (1 - self.geom_prob) ** k
        else:
            self.K_pmf = lambda k: self.geom_prob * (1 - self.geom_prob) ** k / (
                    1 - (1 - self.geom_prob) ** (self.K_cap + 1))
            self.K_dist = Categorical(probs=self.K_pmf(torch.arange(self.K_cap + 1)))
            self.K_reverse_cdf = lambda k: 1 - (1 - (1 - self.geom_prob) ** k) / (
                    1 - (1 - self.geom_prob) ** (self.K_cap + 1))

        self._update_K()

        self.estimator_fn = None

    def __call__(self, density, x, update_K=True):
        if update_K:
            self._update_K()

        self.get_log_w(density, x, self.num_importance_samples)

        self.p_result = self.estimator_fn(log_w=self.log_w, K=self.K)  # shape (log_p.shape[0], 1)
        self.p_loss = self.p_result.mean()

        self.q_loss = self._compute_q_loss_unbiased(density, x)

        return {"losses": {"p": self.p_loss, "q": self.q_loss}}

    def _update_K(self):
        self.K = int(self.K_dist.sample().item())
        self.num_importance_samples = 2 ** (self.K + 1 + self.I0_start_level)

    def _compute_q_loss_unbiased(self, density, x):
        if self.q_train_method == "var":
            q_loss = self.p_loss.square()
        elif self.q_train_method == "unbiased rws":
            q_loss = - self.p_loss
        elif self.q_train_method == "unbiased rws_stl":
            train_function = partial(_compute_q_loss, q_train_method=self.q_train_method.split()[1])
            q_loss = self.estimator_fn(log_w=self.log_w, K=self.K, train_function=train_function).mean()
        elif self.q_train_method == "elbo":
            q_loss = - self.log_w.mean()
        elif self.q_train_method in ("iwae", "rws", "rws_stl"):
            self.get_log_w(density, x, self.q_num_importance_samples, tempered=False)
            q_loss = _compute_q_loss(self.log_w, self.q_train_method).mean()
        else:
            assert False, f"Invalid q train method `{self.q_train_method}'"
        return q_loss

    def get_log_w(self, density, x, num_importance_samples, **kwargs):
        self.log_p, self.log_q = compute_log_p_q(density, x, num_importance_samples,
                                                 **get_detach_config(self.q_train_method))
        self.log_w = self.log_p - self.log_q
        return self.log_w, self.log_p, self.log_q


class ML_LL(UnbiasedEstimator):
    def __init__(self, estimator_name, geom_prob, K_cap, I0_start_level, q_train_method, q_num_importance_samples):
        super(ML_LL, self).__init__(geom_prob, K_cap, I0_start_level, q_train_method, q_num_importance_samples)
        if estimator_name == "ss":
            self.estimator_fn = partial(_ss, K_pmf=self.K_pmf, I0_start_level=self.I0_start_level)
        elif estimator_name == "rr":
            self.estimator_fn = partial(_rr, K_reverse_cdf=self.K_reverse_cdf, I0_start_level=self.I0_start_level)
        else:
            assert False, f"Invalid estimator name `{estimator_name}'"


def _ss(log_w, K_pmf, K, I0_start_level, train_function=_iwae):
    log_w = log_w.view(log_w.shape[0], 2 ** (K + 1), 2 ** I0_start_level, 1)
    I0 = train_function(log_w).mean(dim=1)

    log_w = log_w.view(log_w.shape[0], 2, 2 ** (K + I0_start_level), 1)
    lower_level_term = train_function(log_w).mean(dim=1)

    log_w = log_w.view(log_w.shape[0], 2 ** (K + I0_start_level + 1), 1)
    upper_level_term = train_function(log_w)

    ml_ll_ss = I0 + (upper_level_term - lower_level_term) / K_pmf(K)

    return ml_ll_ss


def _rr(log_w, K_reverse_cdf, K, I0_start_level, train_function=_iwae, delta_ks_last_dim=1):
    log_w = log_w.view(log_w.shape[0], 2 ** K, 2 ** (I0_start_level + 1), 1)
    I0 = train_function(log_w).mean(dim=1)

    if K == 0:
        ml_ll_rr = I0
    else:
        delta_ks = torch.empty((log_w.shape[0], K, delta_ks_last_dim), device=log_w.device)
        for k in range(1, K + 1):
            log_w = log_w.view(log_w.shape[0], 2 ** (K - k), 2, 2 ** (I0_start_level + k), 1)
            lower_level_term = train_function(log_w).mean((1, 2))  # shape (x.shape[0], 1)

            log_w = log_w.view(log_w.shape[0], 2 ** (K - k), 2 ** (I0_start_level + k + 1), 1)
            upper_level_term = train_function(log_w).mean(1)

            delta_ks[:, k - 1, :] = upper_level_term - lower_level_term  # shape (x.shape[0], 1)

        inv_weights = 1 / K_reverse_cdf(torch.arange(1, (K + 1), device=log_w.device).float())

        ml_ll_rr = I0 + torch.sum(delta_ks * inv_weights.view(1, K, 1), 1)

    return ml_ll_rr


class SUMO(UnbiasedEstimator):
    def __init__(self, min_importance_samples, q_train_method, q_num_importance_samples, truncate_alpha=80):
        self.min_importance_samples = min_importance_samples
        self.q_train_method = q_train_method
        self.q_num_importance_samples = q_num_importance_samples
        self.truncate_alpha = truncate_alpha

        # min_importance_samples is similar to I0_start_level in SS/RR, > 1
        # Note that this parameterization of K starts from 0, but is equivalent to SUMO paper
        self.K_inverse_cdf = lambda u: np.floor(1 / (1 - u) - 1) if u < 1 - 1 / truncate_alpha else np.floor(
            (np.log(truncate_alpha) + np.log(1 - u)) / np.log(0.9) + truncate_alpha - 1)
        self.K_reverse_cdf = np.vectorize(
            lambda k: 1 / (k + 1) if k < (truncate_alpha - 1) else 1 / truncate_alpha * 0.9 ** (k - truncate_alpha + 1))

        self._update_K()

        self.estimator_fn = partial(_sumo, K_reverse_cdf=self.K_reverse_cdf,
                                    min_importance_samples=self.min_importance_samples)

        self.p_train_function = _iwae

    def _update_K(self):
        # Inversion sampling.
        self.K = self.K_inverse_cdf(np.random.rand()).astype(int)
        self.num_importance_samples = self.K + self.min_importance_samples + 1


def _sumo(log_w, K_reverse_cdf, K, min_importance_samples, train_function=_iwae):
    assert min_importance_samples == log_w.shape[1] - K - 1
    if train_function == _iwae:
        cum_iwae = torch.logcumsumexp(log_w, dim=1) - \
                   torch.arange(1, K + min_importance_samples + 2, device=log_w.device).float().log().view(
                       (1, K + min_importance_samples + 1, 1))

        delta_ks = cum_iwae[:, min_importance_samples:] - cum_iwae[:, (min_importance_samples - 1):-1]

        inv_weights = 1 / torch.from_numpy(K_reverse_cdf(np.arange(K + 1))).float().to(log_w.device)

        sumo = torch.sum(delta_ks * inv_weights.view((1, -1, 1)), dim=1) + cum_iwae[:, min_importance_samples - 1, :]

        return - sumo
    else:
        I0 = train_function(log_w[:, :min_importance_samples])

        delta_ks = torch.empty((log_w.shape[0], K + 1, 1), device=log_w.device)
        for k in range(K + 1):
            lower_level_term = train_function(log_w[:, :(min_importance_samples + k)])
            upper_level_term = train_function(log_w[:, :(min_importance_samples + k + 1)])

            delta_ks[:, k, :] = upper_level_term - lower_level_term  # shape (x.shape[0], 1)

        inv_weights = 1 / torch.from_numpy(K_reverse_cdf(np.arange(K + 1))).float().to(log_w.device)

        sumo = I0 + torch.sum(delta_ks * inv_weights.view(1, K + 1, 1), 1)

        return sumo


class TVO:
    def __init__(self, num_importance_samples, q_train_method, num_partitions, partition_type, log_beta_min):
        self.num_importance_samples = num_importance_samples
        self.q_train_method = q_train_method
        self.num_partitions = num_partitions
        self.partition_type = partition_type
        self.log_beta_min = log_beta_min
        self.partition = _get_partition(self.num_partitions, self.partition_type, self.log_beta_min)
        self.multiplier = torch.zeros_like(self.partition)
        self.multiplier[:-1] = self.partition[1:] - self.partition[:-1]

        def _tvo(log_p, log_q, **kwargs): \
                return torch.sum(self.multiplier.to(log_p.device) * _thermo_covariance(log_p, log_q, self.partition),
                                 dim=-1, keepdim=True)

        self.p_train_function = _tvo
        print(self.partition)

    def __call__(self, density, x, update_K=True):
        self.get_log_w(density, x, self.num_importance_samples)

        p_loss = self.p_train_function(self.log_p, self.log_q).mean()

        if self.q_train_method == "tvo":
            q_loss = p_loss
        elif self.q_train_method in ("iwae", "rws", "rws_stl"):
            self.log_w = self.log_p - self.log_q
            q_loss = _compute_q_loss(self.log_w, self.q_train_method).mean()
        else:
            assert False, f"Invalid q train method `{self.q_train_method}'"

        return {"losses": {"p": p_loss, "q": q_loss}}

    def _update_K(self):
        pass

    def get_log_w(self, density, x, num_importance_samples, tempered=False, **kwargs):
        self.log_p, self.log_q = compute_log_p_q(density, x, num_importance_samples,
                                                 **get_detach_config(self.q_train_method))
        self.log_w = self.log_p - self.log_q
        if tempered:
            return self.log_w * self.partition[-2], self.log_p, self.log_q
        return self.log_w, self.log_p, self.log_q


def _thermo_covariance(log_p, log_q, partition):
    if torch.is_tensor(partition):
        partition = partition.to(log_p.device)
    log_w = log_p - log_q  # shape (x.shape[0], importance_samples, 1)
    heated_log_w = log_w * partition

    heated_normalized_w = lognormexp(heated_log_w, dim=-2).exp()
    thermo_log_pi = partition * log_p + (1 - partition) * log_q

    heated_normalized_w_detached = heated_normalized_w.detach()
    if log_p.shape[-2] == 1:
        correction = 1
    else:
        correction = log_p.shape[-2] / (log_p.shape[-2] - 1)

    covariance_estimator = correction * torch.sum(
        heated_normalized_w_detached *
        (log_w - torch.sum(heated_normalized_w * log_w, dim=-2, keepdim=True)).detach() *
        (thermo_log_pi - torch.sum(heated_normalized_w_detached * thermo_log_pi, dim=-2, keepdim=True)),
        dim=-2)

    loss = - (covariance_estimator + torch.sum(heated_normalized_w_detached * log_w, dim=-2))
    loss -= loss.detach()
    loss += _thermo_simple(log_w, partition).detach()

    return loss


def _thermo_simple(log_w, partition):
    if torch.is_tensor(partition):
        partition = partition.to(log_w.device)
    heated_log_w = log_w * partition

    heated_normalized_w = lognormexp(heated_log_w, dim=-2).exp()

    return - torch.sum(heated_normalized_w * log_w, dim=-2)


def _get_partition(num_partitions, partition_type, log_beta_min, device=None):
    if device is None:
        device = torch.device('cpu')
    if num_partitions == 1:
        partition = torch.tensor([0, 1], dtype=torch.float, device=device)
    else:
        if partition_type == 'linear':
            partition = torch.linspace(0, 1, steps=num_partitions + 1,
                                       device=device)
        elif partition_type == 'log':
            partition = torch.zeros(num_partitions + 1, device=device,
                                    dtype=torch.float)
            partition[1:] = torch.logspace(
                log_beta_min, 0, steps=num_partitions, device=device,
                dtype=torch.float)
    return partition


class ML_TVO(UnbiasedEstimator):
    def __init__(self, estimator_name, geom_prob, K_cap, I0_start_level, p_train_method, q_train_method,
                 q_num_importance_samples, num_partitions, partition_type, log_beta_min):
        super(ML_TVO, self).__init__(geom_prob, K_cap, I0_start_level, q_train_method, q_num_importance_samples)
        self.p_train_method = p_train_method
        self.num_partitions = num_partitions
        self.partition_type = partition_type
        self.log_beta_min = log_beta_min
        self.partition = _get_partition(self.num_partitions, self.partition_type, self.log_beta_min)
        self.multiplier = torch.zeros_like(self.partition)
        self.multiplier[:-1] = self.partition[1:] - self.partition[:-1]
        print(self.partition)

        if estimator_name == "ss":
            self.estimator_fn = partial(_ss, K_pmf=self.K_pmf, I0_start_level=self.I0_start_level)
        elif estimator_name == "rr":
            self.estimator_fn = partial(_rr, K_reverse_cdf=self.K_reverse_cdf, I0_start_level=self.I0_start_level)
        else:
            assert False, f"Invalid estimator name `{estimator_name}'"

        def _ml_tvo(log_w, **kwargs):
            return torch.sum(self.multiplier.to(log_w.device) * _thermo_simple(log_w, self.partition),
                             dim=-1, keepdim=True)

        self.p_train_function = _ml_tvo

    def __call__(self, density, x, update_K=True):
        if update_K:
            self._update_K()

        self.get_log_w(density, x, self.num_importance_samples)

        self.p_loss = self.estimator_fn(log_w=self.log_w, K=self.K, train_function=self.p_train_function).mean()
        self.q_loss = self._compute_q_loss_unbiased(density, x)

        return {"losses": {"p": self.p_loss, "q": self.q_loss}}

    def _compute_q_loss_unbiased(self, density, x):
        if self.q_train_method == "iwae":
            self.get_log_w(density, x, self.q_num_importance_samples, tempered=False)
            q_loss = _compute_q_loss(self.log_w, self.q_train_method).mean()
        else:
            assert False, f"Invalid q train method `{self.q_train_method}'"
        return q_loss

    def get_log_w(self, density, x, num_importance_samples, tempered=False, **kwargs):
        self.log_p, self.log_q = compute_log_p_q(density, x, num_importance_samples,
                                                 **get_detach_config(self.q_train_method))
        self.log_w = self.log_p - self.log_q
        if tempered:
            return self.log_w * self.partition[-2], self.log_p, self.log_q
        return self.log_w, self.log_p, self.log_q

