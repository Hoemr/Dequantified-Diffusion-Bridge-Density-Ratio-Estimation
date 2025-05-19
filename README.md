# Official PyTorch implementation for D3RE

This repo contains a reference implementation for D3RE as described in the paper:
> Title: Dequantified Diffusion Schrödinger Bridge for Density Ratio Estimation </br>
> Conference: International Conference on Machine Learning (ICML), 2025. </br>
> Paper: [https://arxiv.org/abs/2111.11010](https://arxiv.org/abs/2505.05034) </br>

Note that the code structure is a direct extension of: [https://github.com/yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)  and [https://github.com/ermongroup/dre-infinity.git](https://github.com/ermongroup/dre-infinity.git). 

## Introduction
**D3RE is a plug-and-play method for ODE-based density ratio estimation methods.**

The core module is summarized as follows (`sde_lib.py`):
```
class InterpXt(SDE):
    def __init__(self, args=None, N=1000, beta_min=0.1, beta_max=20):
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        
        self.args = args
        self.GRBC_epsilon = args.GRBC_epsilon
        self.bridge = args.bridge
        self.combination_type = args.combination_type  # e.g., "linear", "VPSDE", "cosine"
        self.gamma_t = args.gamma_t
        self.OT = args.OT

        if self.OT:
            self.ot_sampler = OTPlanSampler(method="exact")
            if self.combination_type != "linear":
                raise ValueError(f"OT is {self.OT} and combination_type is {self.combination_type}, not linear")
        else:
            self.ot_sampler = None

    @property
    def T(self):
        return 1.0

    def get_marginal_var_fn(self):
        if self.bridge:
            return lambda t : self.bridge_var(t)
        else:
            return lambda t : self.get_alpha_t_beta_t(t)[1] ** 2

    def get_marginal_var_dt_fn(self):
        if self.bridge:
            return lambda t : self.bridge_dt_var(t)
        else:
            return lambda t : self.get_d_beta_dt(t)

    def bridge_std(self, t, alpha_t=None, beta_t=None):
        return torch.sqrt(self.bridge_var(t, alpha_t=alpha_t, beta_t=beta_t))

    def bridge_dt_std(self, t, alpha_t=None, beta_t=None):
        return 0.5 * self.bridge_dt_var(t, alpha_t=alpha_t, beta_t=beta_t) / self.bridge_var(t, alpha_t=alpha_t, beta_t=beta_t)

    def bridge_var(self, t, alpha_t=None, beta_t=None):
        if alpha_t is None:
            alpha_t, beta_t = self.get_alpha_t_beta_t(t)
        return self.gamma_t * t * (1 - t) + (alpha_t**2 + beta_t**2) * self.GRBC_epsilon

    def bridge_dt_var(self, t, alpha_t=None, beta_t=None):
        if alpha_t is None:
            alpha_t, beta_t = self.get_alpha_t_beta_t(t)
        return self.gamma_t * (1. - 2 * t) + 2 * (alpha_t + beta_t) * self.GRBC_epsilon

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        N = np.prod(z.shape[1:])
        squared_sum = torch.sum(z ** 2, dim=tuple(range(1, z.dim())))
        return -N / 2. * np.log(2 * np.pi) - squared_sum / 2.

    # Methods that require dimension-specific implementations.
    def sde(self, x, t):
        raise NotImplementedError("sde method must be implemented in subclass")

    def discretize(self, x, t):
        raise NotImplementedError("discretize method must be implemented in subclass")

    def get_alpha_t_beta_t(self, t):
        if self.combination_type == "linear":
            alpha_t = 1 - t
            beta_t = t
        elif self.combination_type == "VPSDE":
            log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
            alpha_t = torch.exp(log_mean_coeff)
            beta_t = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        elif self.combination_type == "follmer":
            alpha_t = torch.sqrt(1 - t ** 2)
            beta_t = t
        elif self.combination_type == "cosine":
            s = torch.tensor(0.008, dtype=t.dtype, device=t.device)
            fn_t = torch.cos((t + s) / (1 + s) * math.pi / 2)
            fn_0 = torch.cos(s / (1 + s) * math.pi / 2)
            alpha_t_bar = fn_t / fn_0
            alpha_t = torch.sqrt(alpha_t_bar)
            beta_t = torch.sqrt(1 - alpha_t_bar)
        else:
            raise ValueError(f"Combination type {self.combination_type} not implemented in InterpXt.")
        return alpha_t, beta_t
      
    def get_d_beta_dt(self, t):
        if self.combination_type == "linear":
            d_beta_dt = 1.
        elif self.combination_type == "VPSDE":
            alpha_t, beta_t = self.get_alpha_t_beta_t(t)
            term1 = alpha_t ** 2 / beta_t
            term2 = 0.5 * ( (self.beta_1 - self.beta_0) * t + self.beta_0 )
            d_beta_dt = term1 * term2
        elif self.combination_type == "follmer":
            d_beta_dt = 1.
        elif self.combination_type == "cosine":
            s = torch.tensor(0.008, dtype=t.dtype, device=t.device)
            theta_t = (t + s) / (1 + s) * math.pi / 2
            sin_theta_t = torch.sin(theta_t)
            cos_theta_t = torch.cos(theta_t)
            theta_0 = s / (1 + s) * math.pi / 2
            fn_0 = torch.cos(theta_0)
            alpha_t_bar = cos_theta_t / fn_0
            beta_t = torch.sqrt(1 - alpha_t_bar)
            numerator = sin_theta_t * math.pi
            denominator = 4 * (1 + s) * fn_0 * beta_t
            d_beta_dt = numerator / denominator
        else:
            raise ValueError(f"Combination type {self.combination_type} not implemented in InterpXt.")
        
        return d_beta_dt

    def marginal_sample(self, x0, xT, t):
        if self.OT:
            x0, xT = self.ot_sampler.sample_plan(x0, xT, replace=True)
        alpha_t, beta_t = self.get_alpha_t_beta_t(t)
        xt = alpha_t * x0 + beta_t * xT

        if self.bridge:
            bridge_std = self.bridge_std(t, alpha_t=alpha_t, beta_t=beta_t)
            z = torch.randn_like(x0).to(x0)
            xt = xt + bridge_std * z
        else:
            z = torch.randn_like(x0).to(x0)
            xt = xt + torch.sqrt((alpha_t**2 + beta_t**2) * self.GRBC_epsilon) * z
        return xt

    def inner_product(self, x, y):
        assert x.shape == y.shape, "Input tensor shapes must match."
        dims_to_sum = tuple(range(1, len(x.shape)))
        return torch.sum(x * y, dim=dims_to_sum).view(-1, 1)
```

> For Toy datasets:
```
class ToyInterpXt(InterpXt):
    def __init__(self, args=None, N=1000, beta_min=0.1, beta_max=20):
        super().__init__(args=args, N=N, beta_min=beta_min, beta_max=beta_max)
```

> For Image datasets:
```
class ImageInterpXt(InterpXt):
    def __init__(self, args=None, N=1000, beta_min=0.1, beta_max=20):
        super().__init__(args=args, N=N, beta_min=beta_min, beta_max=beta_max)
    
    def bridge_std(self, t):
        return super().bridge_std(t).view(t.size(0), 1, 1, 1)
    
    def bridge_var(self, t):
        return super().bridge_var(t).view(t.size(0), 1, 1, 1)
      
    def marginal_std_bridge(self, t):
        std = torch.sqrt(t * (self.T - t) / self.T)
        return std.view(t.size(0), 1, 1, 1)

    def get_alpha_t_beta_t(self, t):
        alpha_t, beta_t = super().get_alpha_t_beta_t(t)
        alpha_t = alpha_t.view(t.size(0), 1, 1, 1)
        beta_t = beta_t.view(t.size(0), 1, 1, 1)
        return alpha_t, beta_t
``` 

## For the MI estimation experiments using the joint score matching objective:
> For 40-D, we set `config.data.dim=40, config.training.n_iters=20001, config.training.eval_freq=100`.
- DRE-$\infty$ (baseline)
```
python3 main.py --toy \
--config configs/gmm_mutual_info/joint/param.py --mode=train \
--config.model.type=joint --config.training.joint=True \
--config.data.dim=40 --config.seed=7777 --config.training.n_iters=40001 \
--workdir=./results/gmm_mi_40d_param_joint/ \
--config.training.batch_size=512 --config.training.eval_freq=100 \
--config.training.reweight=True --config.device_id=0
```

- D3RE (ours)
```
python3 main.py --toy \
--config configs/gmm_mutual_info/joint/param.py --mode=train \
--config.model.bridge=True \
--config.model.type=joint --config.training.joint=True \
--config.data.dim=40 --config.seed=7777 --config.training.n_iters=40001 \
--workdir=./results/gmm_mi_40d_param_joint_bridge/ \
--config.training.batch_size=512 --config.training.eval_freq=100 \
--config.training.reweight=True --config.device_id=0
```

> For 80-D, we set `config.data.dim=80`, `config.training.eval_freq=100`, `config.training.n_iters=50001`.

> For 160-D, we set `config.data.dim=160`, `config.training.eval_freq=1000`, and `config.training.n_iters=200001`.

> For 320-D, we set `config.data.dim=320`, `config.training.eval_freq=1000`, `config.training.batch_size=256`, and `config.training.n_iters=400001`.

## For the MNIST experiments:

First, we use the `nsf` codebase to train the flow models. All pre-trained model checkpoints (Gaussian, Copula, RQ-NSF) can be found in `flow_ckpts/`. There is no need to re-train the flow models from scratch and all the time score networks take into account the particular ways that the data has been preprocessed.

> (a) For the Gaussian noise model:
- DRE-$\infty$ (baseline)
```
python3 main.py --flow \
--config configs/mnist/z_gaussian_time_interpolate.py \
--mode=train --doc=z_unet_lin_emb_noise \
--workdir=./results/mnist_z_unet_lin_emb_noise
```

- D3RE (ours)
```
python3 main.py --flow \
--config configs/mnist/z_gaussian_time_interpolate.py \
--config.model.bridge=True \
--mode=train --doc=z_unet_lin_emb_noise_bridge \
--workdir=./results/mnist_z_unet_lin_emb_noise_bridge --config.device_id=3
```


> (b) For the copula:
- DRE-$\infty$ (baseline)
```
python3 main.py --flow \
--config configs/mnist/z_copula_time_interpolate.py \
--mode=train --doc=z_unet_lin_emb_copula \
--config.training.likelihood_weighting=True \
--workdir=./results/mnist_z_unet_lin_emb_copula
```

- D3RE (ours)
```
python3 main.py --flow \
--config configs/mnist/z_copula_time_interpolate.py \
--config.model.bridge=True \
--mode=train --doc=z_unet_lin_emb_copula_bridge \
--config.training.likelihood_weighting=True \
--workdir=./results/mnist_z_unet_lin_emb_copula_bridge --config.device_id=0
```

> (c) For the RQ-NSF flow model:
- DRE-$\infty$ (baseline)
```
python3 main.py --flow \
--config configs/mnist/z_flow_time_interpolate.py \
--mode=train --doc=z_unet_lin_emb_flow \
--config.training.likelihood_weighting=True \
--workdir=./results/mnist_z_unet_lin_emb_flow
```

- D3RE (ours)
```
python3 main.py --flow \
--config configs/mnist/z_flow_time_interpolate.py \
--config.model.bridge=True \
--mode=train --doc=z_unet_lin_emb_flow_bridge \
--config.training.likelihood_weighting=True \
--workdir=./results/mnist_z_unet_lin_emb_flow_bridge --config.device_id=0
```

## References
If you find this work useful in your research, please consider citing the following paper:
```
@article{chen2025dequantified,
  title={Dequantified Diffusion Schr{\"o}dinger Bridge for Density Ratio Estimation},
  author={Chen, Wei and Li, Shigui and Li, Jiacheng and Yang, Junmei and Paisley, John and Zeng, Delu},
  journal={arXiv preprint arXiv:2505.05034},
  year={2025}
}
@article{choi2021density,
  title={Density Ratio Estimation via Infinitesimal Classification},
  author={Choi, Kristy and Meng, Chenlin and Song, Yang and Ermon, Stefano},
  journal={arXiv preprint arXiv:2111.11010},
  year={2021}
}
```
