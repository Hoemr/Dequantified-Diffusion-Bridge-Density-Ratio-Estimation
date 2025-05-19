# Official PyTorch implementation for D3RE

This repo contains a reference implementation for D3RE as described in the paper:
> Title: Dequantified Diffusion Schrödinger Bridge for Density Ratio Estimation </br>
> Conference: International Conference on Machine Learning (ICML), 2025. </br>
> Paper: [https://arxiv.org/abs/2111.11010](https://arxiv.org/abs/2505.05034) </br>

Note that the code structure is a direct extension of: [https://github.com/yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)  and [https://github.com/ermongroup/dre-infinity.git](https://github.com/ermongroup/dre-infinity.git). 

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
