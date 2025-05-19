cd ~/savepath/cw/dre-free

/public/datasets/binzeng/wei/dre-free/

- 后台运行程序：需要加nohup (no hang up)，因为python进程是bash进程的子进程，在bash进程(ssh断开)被杀后python进程也就被杀了，会导致程序停止

/dev/null 2>&1 &               # 不输入到nohup.out文件

`nohup python main.py /dev/null 2>&1 &`

# For the 1-D peaked Gaussian experiments

- base_line:
  `python main.py --toy --config configs/1d_gaussians/time/param.py --mode=train --doc=1d_peaked_gaussians_param_time --workdir=./results/1d_peaked_gaussians/param_time/`

- our_method
  `python main.py --toy --bridge=True --config configs/1d_gaussians/bridge/param.py --mode=train --doc=1d_peaked_gaussians_bridge --workdir=./results/1d_peaked_gaussians/bridge/`

`python main.py --toy --config configs/1d_gaussians/time/param.py --mode=train --config.model.bridge=True --doc=1d_peaked_gaussians_param_time_bridge --workdir=./results/1d_peaked_gaussians/param_time_bridge/`    (在原来的基础上直接改)

结果展示：
true log ratios: -1999990.9         6.887755
est. log ratios: -1996679.354246278 6.90239044693268    （baseline）                       差异：3311     0.01463544693268
est. log ratios: -1979085.892311453 6.87258202370048    （改SDE，权重考虑x0）               差异：20905    0.01517297629952
est. log ratios: -1979184.685633405 6.90231681320305    （改SDE，权重不考虑x0）             差异：20806    0.01456181320305
est. log ratios: -2002246.220182313 6.89636879665996    （改SDE，权重考虑x0，network改动）  差异：2256     0.008545
est. log ratios: -2003002.583057608 6.90147733240758    （改SDE，权重不考虑x0，network改动）差异：3012     0.01372233240758

true log ratios: -1999990.9         6.887755
est. log ratios: -1996679.354246278 6.90239044693268    （baseline）                       
est. log ratios: -1979085.892311453 6.87258202370048    （改SDE，权重考虑x0）               
est. log ratios: -1979184.685633405 6.90231681320305    （改SDE，权重不考虑x0）             
est. log ratios: -2002246.220182313 6.89636879665996    （改SDE，权重考虑x0，network改动）  
est. log ratios: -2003002.583057608 6.90147733240758    （改SDE，权重不考虑x0，network改动）

- middle_states
  `python3 toy_middle_states.py --toy --bridge=True --config configs/1d_gaussians/bridge/param.py --mode=train --doc=1d_peaked_gaussians_bridge --workdir=./results/1d_peaked_gaussians/bridge/`

`python3 toy_middle_states.py --toy --config configs/1d_gaussians/time/param.py --mode=train --doc=1d_peaked_gaussians_param_time --workdir=./results/1d_peaked_gaussians/param_time/`

# For the MI estimation experiments using the joint score matching objective:

注：新的结果把迭代次数都翻倍了，为了得到稳定的MI曲线

## For 40-D, we set `config.data.dim=40, config.training.n_iters=20001, config.training.eval_freq=2000`.

- (baseline)

```
nohup python3 main.py --toy \
--config configs/gmm_mutual_info/joint/param.py --mode=train \
--doc=mi_40d_param_joint --config.model.type=joint \
--config.training.joint=True --config.data.dim=40 \
--config.seed=7777 --config.training.n_iters=40001 \
--workdir=./results/gmm_mi_40d_param_joint/ \
--config.training.batch_size=512 --config.training.eval_freq=100 \
--config.training.reweight=True --config.device_id=0 /dev/null 2>&1 &
```

- (ours)

```
nohup python3 main.py --toy \
--config configs/gmm_mutual_info/joint/param.py --mode=train \
--config.model.bridge=True --doc=mi_40d_param_joint_bridge \
--config.model.type=joint --config.training.joint=True \
--config.data.dim=40 --config.seed=7777 --config.training.n_iters=40001 \
--workdir=./results/gmm_mi_40d_param_joint_bridge/ \
--config.training.batch_size=512 --config.training.eval_freq=100 \
--config.training.reweight=True --config.device_id=0 /dev/null 2>&1 &
```

结果展示：
true MI 10
empirical MI 9.997594205392035
est MI 9.99391927419509       (baseline)
empirical MI 10.072954994381078
est MI 10.048632042262925  (baseline)
empirical MI 10.01517842509937
est MI 10.010235691269306    (ours)

## For 80-D, we set `config.data.dim=80, config.training.n_iters=50001, config.training.eval_freq=5000`.

- baseline

```
nohup python3 main.py --toy \
--config configs/gmm_mutual_info/joint/param.py --mode=train \
--doc=mi_80d_param_joint --config.model.type=joint \
--config.training.joint=True --config.data.dim=80 \
--config.seed=7777 --config.training.n_iters=100001 \
--workdir=./results/gmm_mi_80d_param_joint/ \
--config.training.batch_size=512 --config.training.eval_freq=100 \
--config.training.reweight=True --config.device_id=5 /dev/null 2>&1 &
```

- ours

```
nohup python3 main.py --toy \
--config configs/gmm_mutual_info/joint/param.py --mode=train \
--config.model.bridge=True --doc=mi_80d_param_joint_bridge \
--config.model.type=joint --config.training.joint=True \
--config.data.dim=80 --config.seed=7777 --config.training.n_iters=100001 \
--workdir=./results/gmm_mi_80d_param_joint_bridge/ \
--config.training.batch_size=512 --config.training.eval_freq=100 \
--config.training.reweight=True --config.device_id=5 /dev/null 2>&1 &
```

结果展示：

true MI 20
empirical MI 20.00006936650147
est MI 19.974608554247123       (baseline)
empirical MI 20.02596947274375
est MI 19.83254004233047
empirical MI 20.00036388430718
est MI 19.725821402874057
empirical MI 20.03453551192426
est MI 20.01798742942098     (ours)

true MI 20
empirical MI 20.00006936650147
est MI 19.974608554247123       (baseline)
empirical MI 20.03453551192426
est MI 20.01798742942098     (ours)

## For 160-D, we set `config.data.dim=160, config.training.eval_freq=5000, and config.training.n_iters=200001`.

- baseline

```
nohup python3 main.py --toy \
--config configs/gmm_mutual_info/joint/param.py --mode=train \
--doc=mi_160d_param_joint --config.model.type=joint \
--config.training.joint=True --config.data.dim=160 \
--config.seed=7777 --config.training.n_iters=400001 \
--workdir=./results/gmm_mi_160d_param_joint/ \
--config.training.batch_size=512 --config.training.eval_freq=100 \
--config.training.reweight=True --config.device_id=3 /dev/null 2>&1 &
```

- (ours)

```
nohup python3 main.py --toy \
--config configs/gmm_mutual_info/joint/param.py --mode=train \
--config.model.bridge=True --doc=mi_160d_param_joint_bridge \
--config.model.type=joint --config.training.joint=True \
--config.data.dim=160 --config.seed=7777 --config.training.n_iters=400001 \
--workdir=./results/gmm_mi_160d_param_joint_bridge/ \
--config.training.batch_size=512 --config.training.eval_freq=100 \
--config.training.reweight=True --config.device_id=4 /dev/null 2>&1 &
```

true MI 40
empirical MI 40.14069698702341
est MI 40.07709508350471       (baseline)
empirical MI 39.945750356011025
est MI 39.87996479035554
empirical MI 40.004949343860304
est MI 39.916492489059614
empirical MI 39.84716836620967
est MI 39.71257259600674
empirical MI 39.99135479100257
est MI 39.945757855109186    (ours)
true MI 40
empirical MI 40.058904832833676
est MI 40.01396614736603

## For 320-D, we set `config.data.dim=320, config.training.eval_freq=8000, config.training.batch_size=256, and config.training.n_iters=400001`.

- baseline

```
nohup python3 main.py --toy \
--config configs/gmm_mutual_info/joint/param.py --mode=train \
--doc=mi_320d_param_joint --config.model.type=joint \
--config.training.joint=True --config.data.dim=320 \
--config.seed=7777 --config.training.n_iters=500001 \
--workdir=./results/gmm_mi_320d_param_joint/ \
--config.training.batch_size=256 --config.training.eval_freq=2000 \
--config.training.reweight=True --config.device_id=1 /dev/null 2>&1 &
```

- ours

```
nohup python3 main.py --toy \
--config configs/gmm_mutual_info/joint/param.py --mode=train \
--config.model.bridge=True --doc=mi_320d_param_joint_bridge \
--config.model.type=joint --config.training.joint=True \
--config.data.dim=320 --config.seed=7777 --config.training.n_iters=500001 \ 
--workdir=./results/gmm_mi_320d_param_joint_bridge/ \
--config.training.batch_size=256 --config.training.eval_freq=2000 \
--config.training.reweight=True --config.device_id=2 /dev/null 2>&1 &
```

nohup python3 main.py --toy \
--config configs/gmm_mutual_info/joint/param.py --mode=train \
--doc=mi_320d_param_joint_2 --config.model.type=joint \
--config.training.joint=True --config.data.dim=320 \
--config.seed=7777 --config.training.n_iters=500001 \
--workdir=./results/gmm_mi_320d_param_joint_2/ \
--config.training.batch_size=256 --config.training.eval_freq=8000 \
--config.training.reweight=True --config.device_id=0 /dev/null 2>&1 &

nohup python3 main.py --toy \
--config configs/gmm_mutual_info/joint/param.py --mode=train \
--config.model.bridge=True --doc=mi_320d_param_joint_bridge_2 \
--config.model.type=joint --config.training.joint=True \
--config.data.dim=320 --config.seed=7777 --config.training.n_iters=500001 \
--workdir=./results/gmm_mi_320d_param_joint_bridge_2/ \
--config.training.batch_size=256 --config.training.eval_freq=8000 \
--config.training.reweight=True --config.device_id=5 /dev/null 2>&1 &

true MI 80
empirical MI 80.01590395299611
est MI 80.7966808494788
empirical MI 80.05440353607027
est MI 80.82907903854006
empirical MI 80.02330176121225
est MI 80.67495356927617
empirical MI 80.09209845332815
est MI 80.60350249544598

## Toy2D

- baseline  (PID:40170, 45108)

```
nohup python3 main.py --toy \
--config configs/toy2d/joint/param.py --mode=train \
--doc=toy2d_param_joint --config.model.type=time \
--config.training.joint=False --config.seed=7777 --config.training.n_iters=2000001 \
--workdir=./results/toy2d_param_joint_checkerboard_8gaussians/ \
--config.training.batch_size=2048 --config.training.eval_freq=500 \
--config.data.toy_type[0]='checkerboard' --config.data.toy_type[1]='8gaussians' \
--config.training.reweight=True --config.device_id=3 /dev/null 2>&1 &
```

```
nohup python3 main.py --toy \
--config configs/toy2d/joint/param.py --mode=train \
--doc=toy2d_param_time --config.model.type=time\
--config.training.joint=False --config.seed=7777 --config.training.n_iters=2000001 \
--workdir=./results/toy2d_param_time_checkerboard_8gaussians/ \
--config.training.batch_size=2048 --config.training.eval_freq=500 \
--config.data.toy_type[0]='checkerboard' --config.data.toy_type[1]='8gaussians' \
--config.training.reweight=True --config.device_id=3 /dev/null 2>&1 &
```

- ours (PID:42787, 45427)

```
nohup python3 main.py --toy \
--config configs/toy2d/joint/param.py --mode=train \
--doc=toy2d_param_joint_bridge --config.model.type=time --config.model.bridge=True \
--config.training.joint=False --config.seed=7777 --config.training.n_iters=2000001 \
--workdir=./results/toy2d_param_joint_bridge_checkerboard_8gaussians/ \
--config.training.batch_size=2048 --config.training.eval_freq=500 \
--config.data.toy_type[0]='checkerboard' --config.data.toy_type[1]='8gaussians' \
--config.training.reweight=True --config.device_id=2 /dev/null 2>&1 &
```

```
nohup python3 main.py --toy \
--config configs/toy2d/joint/param.py --mode=train \
--doc=toy2d_param_time_bridge --config.model.type=time --config.model.bridge=True \
--config.training.joint=False --config.seed=7777 --config.training.n_iters=2000001 \
--workdir=./results/toy2d_param_time_bridge_checkerboard_8gaussians/ \
--config.training.batch_size=2048 --config.training.eval_freq=500 \
--config.data.toy_type[0]='checkerboard' --config.data.toy_type[1]='8gaussians' \
--config.training.reweight=True --config.device_id=2 /dev/null 2>&1 &
```

## For the MNIST experiments:

- (a) For the Gaussian noise model:

- - baseline

python3 main.py --flow \
--config configs/mnist/z_gaussian_time_interpolate.py \
--mode=train --doc=z_unet_lin_emb_noise \
--workdir=./results/mnist_z_unet_lin_emb_noise --config.device_id=3

nohup python3 main.py --flow --config configs/mnist/z_gaussian_time_interpolate.py --mode=train --doc=z_unet_lin_emb_noise --workdir=./results/mnist_z_unet_lin_emb_noise --config.device_id=3 /dev/null 2>&1 &

- - ours

python3 main.py --flow \
--config configs/mnist/z_gaussian_time_interpolate.py \
--config.model.bridge=True \
--mode=train --doc=z_unet_lin_emb_noise_bridge \
--workdir=./results/mnist_z_unet_lin_emb_noise_bridge --config.device_id=3

nohup python3 main.py --flow \
--config configs/mnist/z_gaussian_time_interpolate.py \
--config.model.bridge=True \
--mode=train --doc=z_unet_lin_emb_noise_bridge \
--workdir=./results/mnist_z_unet_lin_emb_noise_bridge --config.device_id=3 /dev/null 2>&1 &

nohup python3 main.py --flow \

--config configs/mnist/z_gaussian_time_interpolate.py \

--config.model.bridge=True --config.training.n_iters=400001 \
--mode=train --doc=z_unet_lin_emb_noise_bridge \
--workdir=./results/mnist_z_unet_lin_emb_noise_bridge --config.device_id=3 /dev/null 2>&1 &

- (b) For the copula:

- - baseline

python3 main.py --flow \
--config configs/mnist/z_copula_time_interpolate.py \
--mode=train --doc=z_unet_lin_emb_copula \
--config.training.likelihood_weighting=True \
--workdir=./results/mnist_z_unet_lin_emb_copula --config.device_id=2

nohup python3 main.py --flow \
--config configs/mnist/z_copula_time_interpolate.py \
--mode=train --doc=z_unet_lin_emb_copula \
--config.training.likelihood_weighting=True \
--workdir=./results/mnist_z_unet_lin_emb_copula --config.device_id=2 /dev/null 2>&1 &

- - ours

python3 main.py --flow \
--config configs/mnist/z_copula_time_interpolate.py \
--config.model.bridge=True \
--mode=train --doc=z_unet_lin_emb_copula_bridge \
--config.training.likelihood_weighting=True \
--workdir=./results/mnist_z_unet_lin_emb_copula_bridge --config.device_id=4

nohup python3 main.py --flow \
--config configs/mnist/z_copula_time_interpolate.py \
--config.model.bridge=True \
--mode=train --doc=z_unet_lin_emb_copula_bridge \
--config.training.likelihood_weighting=True \
--workdir=./results/mnist_z_unet_lin_emb_copula_bridge --config.device_id=4 /dev/null 2>&1 &

- (c) For the RQ-NSF flow model:

- - baseline

python3 main.py --flow \
--config configs/mnist/z_flow_time_interpolate.py \
--mode=train --doc=z_unet_lin_emb_flow \
--config.training.likelihood_weighting=True \
--workdir=./results/mnist_z_unet_lin_emb_flow --config.device_id=1

nohup python3 main.py --flow \
--config configs/mnist/z_flow_time_interpolate.py \
--mode=train --doc=z_unet_lin_emb_flow \
--config.training.likelihood_weighting=True \
--workdir=./results/mnist_z_unet_lin_emb_flow --config.device_id=1 /dev/null 2>&1 &

- - ours

python3 main.py --flow \
--config configs/mnist/z_flow_time_interpolate.py \
--config.model.bridge=True \
--mode=train --doc=z_unet_lin_emb_flow_bridge \
--config.training.likelihood_weighting=True \
--workdir=./results/mnist_z_unet_lin_emb_flow_bridge --config.device_id=5

nohup python3 main.py --flow \
--config configs/mnist/z_flow_time_interpolate.py \
--config.model.bridge=True \
--mode=train --doc=z_unet_lin_emb_flow_bridge \
--config.training.likelihood_weighting=True \
--workdir=./results/mnist_z_unet_lin_emb_flow_bridge --config.device_id=5 /dev/null 2>&1 &
