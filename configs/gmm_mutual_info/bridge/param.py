from configs.default_bridge_toy_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.joint = False
  training.n_iters = 20000
  training.batch_size = 512
  training.reweight = True

  # data
  data = config.data
  data.dataset = 'GaussiansforMI'
  data.eps = 1e-5
  data.dim = 40  # [40, 80, 160, 320]
  data.rho = 0.8
  data.sigmas = [0.001, 1.]
  data.centered = False

  # model
  model = config.model
  model.name = 'toy_bridge_scorenet'
  model.nf = 64
  
  # parameters of bridge model
  training.reweight = True
  training.resume = False
  
  data.approximate = False
  data.gaussian = True
  
  model.lambda_square = 30
  model.N = 100
  model.schedule = "cosine"    # linear, cosine
  model.eps = 0.005
  model.rtol = 1e-3
  model.atol = 1e-3
  model.method = 'RK45'
  model.integral_eps = 5e-4
  model.integral_mode = "int_f_only"     # "int_f_only"     "int_both"

  return config
