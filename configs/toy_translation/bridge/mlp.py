from configs.default_bridge_toy_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.joint = False

  # data
  data = config.data
  data.dataset = '8gaussians'
  data.eps = 1e-5
  data.dim = 2
#   data.sigmas = [0.001, 1.]
  data.centered = False

  # model
  model = config.model
  model.type = 'time'
  model.param = False
  model.name = 'toy_time_scorenet'
  model.nf = 64

  return config
