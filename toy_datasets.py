import math
import torch
import numpy as np
from torch.distributions import MultivariateNormal, Normal

from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_dataset(config, sde=None):
    if config.data.dataset == 'GMM':
        return GMMDist(config.data.dim, config.device)
    elif config.data.dataset == 'ToyGMM':
        return ToyGMM(sde, config.data.mean_q, config.data.mean_p,
                      config.data.dim)
    elif config.data.dataset == 'GaussiansforMI':
        return GaussiansforMI(config.data.dim, config.device)
    elif config.data.dataset == 'PeakedGaussians':
        return PeakedGaussians(config.data.dim, config.data.sigmas, config.device)
    elif config.data.dataset == 'Toy2D':
        return Toy2D(config.data.dim, config.data.toy_type, config.device)
    else:
        raise NotImplementedError
    

class Toy2D(object):
    def __init__(self, dim, toy_type, device):
        self.dim = dim
        self.device = device
        self.toy_type = toy_type

        self.q = toy_type[0]
        self.p = toy_type[1]
        
    def inf_train_gen(self, toy_type, rng=None, batch_size=200):
        if rng is None:
            rng = np.random.RandomState()

        if toy_type == "swissroll":
            data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
            data = data.astype("float32")[:, [0, 2]]
            data /= 5
            return data

        elif toy_type == "circles":
            data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
            data = data.astype("float32")
            data *= 3
            return data

        elif toy_type == "rings":
            n_samples4 = n_samples3 = n_samples2 = batch_size // 4
            n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

            # so as not to have the first point = last point, we set endpoint=False
            linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
            linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
            linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
            linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

            circ4_x = np.cos(linspace4)
            circ4_y = np.sin(linspace4)
            circ3_x = np.cos(linspace4) * 0.75
            circ3_y = np.sin(linspace3) * 0.75
            circ2_x = np.cos(linspace2) * 0.5
            circ2_y = np.sin(linspace2) * 0.5
            circ1_x = np.cos(linspace1) * 0.25
            circ1_y = np.sin(linspace1) * 0.25

            X = np.vstack([
                np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
                np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
            ]).T * 3.0
            X = util_shuffle(X, random_state=rng)

            # Add noise
            X = X + rng.normal(scale=0.08, size=X.shape)

            return X.astype("float32")

        elif toy_type == "moons":
            data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
            data = data.astype("float32")
            data = data * 2 + np.array([-1, -0.2])
            return data

        elif toy_type == "8gaussians":
            scale = 4.
            centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                    (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                            1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
            centers = [(scale * x, scale * y) for x, y in centers]

            dataset = []
            for i in range(batch_size):
                point = rng.randn(2) * 0.5
                idx = rng.randint(8)
                center = centers[idx]
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            dataset /= 1.414
            return dataset

        elif toy_type == "pinwheel":
            radial_std = 0.3
            tangential_std = 0.1
            num_classes = 5
            num_per_class = batch_size // 5
            rate = 0.25
            rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

            features = rng.randn(num_classes*num_per_class, 2) \
                * np.array([radial_std, tangential_std])
            features[:, 0] += 1.
            labels = np.repeat(np.arange(num_classes), num_per_class)

            angles = rads[labels] + rate * np.exp(features[:, 0])
            rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
            rotations = np.reshape(rotations.T, (-1, 2, 2))

            return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

        elif toy_type == "2spirals":
            n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
            d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
            d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
            x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
            x += np.random.randn(*x.shape) * 0.1
            return x

        elif toy_type == "checkerboard":
            x1 = np.random.rand(batch_size) * 4 - 2
            x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
            x2 = x2_ + (np.floor(x1) % 2)
            return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

        elif toy_type == "line":
            x = rng.rand(batch_size) * 5 - 2.5
            y = x
            return np.stack((x, y), 1)
        elif toy_type == "cos":
            x = rng.rand(batch_size) * 5 - 2.5
            y = np.sin(x) * 2.5
            return np.stack((x, y), 1)
        else:
            raise ValueError("Invalid toy dataset type")
            #return self.inf_train_gen("8gaussians", rng, batch_size)
    
    def sample_sequence_on_the_fly(self, px, qx, t):
        # note: t is functioning as \alpha(t) here
        return torch.sqrt(1 - t**2) * px + (t * qx)

    def sample(self, n, t):
        qx = self.inf_train_gen(self.q, batch_size=n)
        px = self.inf_train_gen(self.p, batch_size=n)
        qx = torch.from_numpy(qx).to(torch.double)
        px = torch.from_numpy(px).to(torch.double)
        xt = self.sample_sequence_on_the_fly(px, qx, t)

        return px.to(self.device), qx.to(self.device), xt.to(self.device)
    
    def sample_marginal(self, n):
        qx = self.inf_train_gen(self.q, batch_size=n)
        px = self.inf_train_gen(self.p, batch_size=n)
        qx = torch.from_numpy(qx).to(torch.double)
        px = torch.from_numpy(px).to(torch.double)  # .to(torch.double)
        return px.to(self.device), qx.to(self.device)


class PeakedGaussians(object):
    """
    The ratio we are estimating is: r(x) = log q(x) - log p(x)
    where q(x) = N(0, 1e-6) and p(x) = N(0, 1)
    q(x) corresponds to T = 1, p(x) corresponds to T = 0
    """
    def __init__(self, dim, sigmas, device):
        self.means = [0, 0]
        self.sigmas = sigmas
        self.dim = dim
        self.device = device

        self.q = Normal(0, self.sigmas[0])
        self.p = Normal(0, self.sigmas[1])

    def sample_sequence_on_the_fly(self, px, qx, t):
        # note: t is functioning as \alpha(t) here
        return torch.sqrt(1 - t**2) * px + (t * qx)

    def sample(self, n, t):
        qx = torch.randn((n, self.dim)) * self.sigmas[0]
        px = torch.randn((n, self.dim)) * self.sigmas[1]
        xt = self.sample_sequence_on_the_fly(px, qx, t)

        return px.to(self.device), qx.to(self.device), xt.to(self.device)

    def log_density_ratios(self, samples):
        log_p = self.p.log_prob(samples)
        log_q = self.q.log_prob(samples)
        log_ratios = log_q - log_p

        return log_ratios

    def log_prob(self, samples, t):
        # get exact form of gaussian
        mu = 0.
        var = (1-t**2) * (self.sigmas[1]**2) + (t**2) * (self.sigmas[0]**2)

        log_q = -((samples - mu) ** 2).sum(dim=-1) / (2 * var) - 0.5 * \
                torch.log(2 * math.pi * var)
        return log_q


# @title Define GMM dataset object
class ToyGMM(object):
    """
    The ratio we are estimating is: r(x) = log q(x) - log p(x)
    where q(x) = N(4, I) and p(x) = N(0, I)
    q(x) corresponds to T = 1 (data), p(x) corresponds to T = 0 (noise)
    """

    def __init__(self, sde, mean_q, mean_p, dim):
        self.sigmas = [1, 1]
        self.means = [mean_q, mean_p]
        self.dim = dim
        self.sde = sde

        self.q = Normal(self.means[0], 1)
        self.p = Normal(self.means[1], 1)

    def sample_sequence_on_the_fly(self, px, qx, t):
        # note: t is functioning as \alpha(t) here
        mean, std = self.sde.marginal_prob(qx, t)  # qx is data
        xt = (torch.randn((len(t), 1)) * std + mean).view(-1, 1)

        return xt

    def sample(self, n, t):
        qx = self.q.sample((n, self.dim))
        px = self.p.sample((n, self.dim))
        xt = self.sample_sequence_on_the_fly(px, qx, t)

        return px, qx, xt

    def sample_data(self, n):
        return self.q.sample((n, self.dim))

    def log_density_ratios(self, samples):
        log_p = self.p.log_prob(samples)
        log_q = self.q.log_prob(samples)
        log_ratios = log_q.sum(-1, keepdims=True) - log_p.sum(-1, keepdims=True)

        return log_ratios

    def log_prob(self, x, t, std=1):
        mean_t, std_t = self.sde.marginal_prob(
            self.means[0] * torch.ones_like(t), t)
        # marginal dist rescales the mean
        std_t = torch.sqrt(
            std_t ** 2 + (std ** 2 * mean_t ** 2 / self.means[0] ** 2))
        log_q = -((x - mean_t) ** 2).sum(-1, keepdims=True) / (
                2 * std_t ** 2) - 0.5 * torch.log(2 * math.pi * std_t ** 2)
        assert log_q.size() == mean_t.size()
        return log_q

    # NOTE: this is the function that's being used for ratio computation
    def log_prob_mixture(self, x, t, batch):
        """
        x: samples (xt)
        t: time
        samples: batch
        """
        mu, sigma = self.sde.marginal_prob(batch, t)
        log_qs = []
        for i in range(len(mu)):
            log_q = (-((x[i] - mu[i]) ** 2) / (
                    2 * sigma[i] ** 2) - 0.5 * torch.log(
                2 * math.pi * sigma[i] ** 2)) + math.log(1. / len(mu))
            log_qs.append(log_q)
        log_q = torch.logsumexp(torch.stack(log_qs, dim=0), dim=0)
        return log_q


class GMMDist(object):
    """
    The ratio we are estimating is: r(x) = log q(x) - log p(x)
    where q(x) = N(4, I) and p(x) = N(0, I)
    q(x) corresponds to T = 1 (data), p(x) corresponds to T = 0 (noise)
    """
    def __init__(self, dim, device):
        self.means = [4, 0]  # 0: q, 1: p
        self.sigmas = [1, 1]
        self.dim = dim
        self.device = device

        self.q = Normal(self.means[0], 1)
        self.p = Normal(self.means[1], 1)

    def sample_sequence_on_the_fly(self, px, qx, t):
        # note: t is functioning as \alpha(t) here
        return torch.sqrt(1 - t**2) * px + (t * qx)

    def sample(self, n, t):
        qx = self.q.sample((n, self.dim))
        px = self.p.sample((n, self.dim))
        xt = self.sample_sequence_on_the_fly(px, qx, t)

        return px.to(self.device), qx.to(self.device), xt.to(self.device)

    def log_density_ratios(self, samples):
        log_p = self.p.log_prob(samples)
        log_q = self.q.log_prob(samples)
        log_ratios = log_q - log_p

        return log_ratios

    def log_prob(self, samples, t, sigma=1):
        # get exact form of gaussian
        mu = (self.means[0] * t) + (self.means[1] * torch.sqrt(1-t**2)).to(
            samples.device)
        # sigma still remains 1 (HACK)
        log_q = -((samples - mu) ** 2).sum(dim=-1) / (2 * sigma ** 2) - 0.5 * np.log(2 * np.pi * sigma ** 2)
        return log_q


class GaussiansforMI(object):
    """
    The ratio we are estimating is: r(x) = log q(x) - log p(x)
    where q(x) = N(0, 1e-6) and p(x) = N(0, 1)
    q(x) corresponds to T = 1, p(x) corresponds to T = 0

    some code adapted from: https://github.com/benrhodes26/tre_code/blob/master/data_handlers/gaussians.py
    """
    def __init__(self, dim, device):
        self.means = [0, 0]
        self.dim = dim
        self.true_mutual_info = self.get_true_mi()
        self.rho = self.get_rho_from_mi(self.true_mutual_info, self.dim)  # correlation coefficient
        self.rhos = np.ones(self.dim // 2) * self.rho
        self.variances = np.ones(self.dim)
        self.cov_matrix = block_diag(*[[[1, self.rho], [self.rho, 1]] for _ in range(self.dim // 2)])
        self.denom_cov_matrix = np.diag(self.variances)
        self.device = device

    @staticmethod
    def get_mi_from_rho(self):
        return -0.5 * np.log(1 - self.rho**2) * self.dim

    @staticmethod
    def get_rho_from_mi(mi, n_dims):
        """Get correlation coefficient from true mutual information"""
        x = (4 * mi) / n_dims  # wtf??
        # x = (2 * mi) / n_dims
        return (1 - np.exp(-x)) ** 0.5  # correlation coefficient

    def get_true_mi(self):
        if self.dim == 20:
            mi = 5
        elif self.dim == 40:
            mi = 10
        elif self.dim == 80:
            mi = 20
        elif self.dim == 160:
            mi = 40
        elif self.dim == 320:
            mi = 80
        else:
            raise NotImplementedError
        return mi

    def sample_gaussian(self, n_samples, cov_matrix):
        prod_of_marginals = multivariate_normal(mean=np.zeros(self.dim),
                                                cov=cov_matrix)
        return prod_of_marginals.rvs(n_samples)

    def sample_data(self, n_samples):
        # p_0 (correlated distribution) -> q(x)
        return torch.from_numpy(
            self.sample_gaussian(n_samples, self.cov_matrix)).float()

    def sample_denominator(self, n_samples):
        # p_m (noise distribution) -> p(x)
        return torch.from_numpy(
            self.sample_gaussian(n_samples, self.denom_cov_matrix)).float()

    def sample_sequence_on_the_fly(self, px, qx, t):
        # note: t is functioning as \alpha(t) here
        return torch.sqrt(1 - t**2) * px + (t * qx)

    def sample(self, n, t):
        qx = self.sample_data(n)  # p_0(x)
        px = self.sample_denominator(n)  # p_m(x)
        xt = self.sample_sequence_on_the_fly(px, qx, t).float()

        # return noise, data, interp
        return px.to(self.device), qx.to(self.device), xt.to(self.device)

    def numerator_log_prob(self, u):
        bivariate_normal = multivariate_normal(
            mean=np.zeros(self.dim), cov=self.cov_matrix)
        log_probs = bivariate_normal.logpdf(u)
        return log_probs

    def denominator_log_prob(self, u):
        prod_of_marginals = multivariate_normal(
            mean=np.zeros(self.dim), cov=self.denom_cov_matrix)
        return prod_of_marginals.logpdf(u)

    def empirical_mutual_info(self, samples=None):
        if samples is None:
            samples = self.sample_data(100000)
        return np.mean(self.numerator_log_prob(samples) - self.denominator_log_prob(samples))