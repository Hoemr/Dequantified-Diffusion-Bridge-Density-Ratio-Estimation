from configs.default_toy_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.joint = False
  training.n_iters = 2000000
  training.batch_size = 2048
  training.reweight = True

  # data
  data = config.data
  data.dataset = 'Toy2D'
  data.toy_type = ['checkerboard', '8gaussians']
  data.eps = 1e-5
  data.dim = 2 
  #data.rho = 0.8
  #data.sigmas = [0.001, 1.]
  data.centered = False

  # model
  model = config.model
  model.type = 'time' # 'joint'
  model.param = False
  model.name = 'toy_scorenet'
  model.nf = 64
  model.z_dim = 64    # 神经网络中间层的维度
  model.bridge = False

  # optimizer
  optim = config.optim
  optim.lr = 1e-3
  optim.scheduler = True

  return config
