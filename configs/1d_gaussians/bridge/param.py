from configs.default_bridge_toy_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.joint = False

  # data
  data = config.data
  data.dataset = 'PeakedGaussians'
  data.eps = 1e-5
  data.dim = 1
  data.sigmas = [0.001, 1.]
  data.centered = False

  # model
  model = config.model
  model.name = 'toy_bridge_time_scorenet'
  model.nf = 64
  
  # parameters of bridge model
  training.reweight = True
  training.resume = False
  training.joint = False
  
  # data.approximate = False
  # data.gaussian = True
  
  # model.lambda_square = 30
  model.N = 100
  # model.schedule = "linear"    # linear, cosine, constant
  model.eps = 0.005
  model.rtol = 1e-3
  model.atol = 1e-3
  model.method = 'RK45'
  model.integral_eps = 1e-5
  # model.integral_mode = "int_both"     # int_f_only, int_both

  return config
