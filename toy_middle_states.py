from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf
import wandb
from tqdm import tqdm
import torch
from datetime import datetime

from bridge_model import toy_datasets
from bridge_model.models import sde_lib 
from bridge_model.models import utils as mutils
from bridge_model.utils import save_checkpoint, restore_checkpoint
from bridge_model.models.toy_networks import ToyBridgeScoreNetwork, ToyBridgeTimeScoreNetwork

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_string("doc", None, "exp_name")
flags.DEFINE_bool("toy", False, "whether to run toy experiment")
flags.DEFINE_bool("bridge", False, "whether to run bridge experiment")
flags.DEFINE_bool("flow", False, "whether to encode the data into latent space using a pre-trained flow")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
  config = FLAGS.config
  workdir = FLAGS.workdir

  # Initialize model.
  # sde = sde_lib.BrownianBrigde(N=config.model.N, device=config.device)
  # score_model = ToyBridgeTimeScoreNetwork(config)
  # score_model = score_model.to(config.device)
  
  sde = sde_lib.ToyInterpXt()
  score_model = mutils.create_model(config, name=config.model.name)
  score_model = score_model.to(config.device)
  
  ema = None

  # Build data iterators
  train_ds = toy_datasets.get_dataset(config)

  # restore checkpoint   
  # checkpoint_dir = os.path.join(workdir, "checkpoints")
  # checkpoint_dir_best = os.path.join(checkpoint_dir, 'best_ckpt.pth')
  checkpoint_dir = os.path.join(workdir, "checkpoints-meta")
  checkpoint_dir_best = os.path.join(checkpoint_dir, "checkpoint.pth")
  loaded_state = torch.load(checkpoint_dir_best, map_location=config.device)
  score_model.load_state_dict(loaded_state['model'], strict=False)
  # score_fn = sde.get_appr_score_fn(score_model, train=False, joint=config.training.joint)

  save_path = os.path.join(workdir, "middle_states")
  tf.io.gfile.makedirs(save_path)
  
  # generate middle states
  batch_size = config.training.batch_size
  num_samples = 50000
  middle_states_num = 10
  val_freq = config.model.N // middle_states_num

  middle_states = {str(i): [] for i in range(middle_states_num+2)}
  # middle_states_true = {str(i): [] for i in range(middle_states_num+1)}
  for batch_index in tqdm(range(int(num_samples/batch_size))):
      cond, batch = train_ds.bridge_sample(batch_size)
      cond = cond.to(config.device)
      batch = batch.to(config.device)
      
      middle_states[str(middle_states_num+1)].append(cond.cpu().view(batch_size,-1))
      # middle_states_true[str(middle_states_num)].append(cond.cpu().view(batch_size,-1))
      xt = cond
      delta_t = torch.tensor(-1).to(xt)
      t = torch.linspace(config.model.integral_eps, 1-config.model.integral_eps,middle_states_num)
      for index, t_i in enumerate(t):
        t_i = torch.ones((batch.shape[0], 1)).to(batch.device) * t_i
        xt = sde.marginal_sample(batch, cond, t_i * sde.T)
        
        # time_score_ti = score_model(xt, cond, t_i * sde.T)
        time_score_ti = score_model(xt, t_i * sde.T)
        
        middle_states[str(index+1)].append(time_score_ti.cpu().view(batch_size,-1))
      
      # for n in range(config.model.N-1, 0, -1):
      #     # approximated xt_1
      #     drift, diffusion = sde.reverse_sde(xt, cond, n, score_fn)
      #     xt_1 = xt + drift * delta_t + diffusion * (torch.sqrt(-delta_t) * torch.randn_like(xt)).to(xt)
          
      #     # true xt_1
      #     xt_1_true = sde.marginal_sample(batch, cond, n/config.model.N*sde.T)
          
      #     # if n % val_freq == 0: # and len(middle_states[str(n // val_freq)])<=batch_index:
      #     #     middle_states[str(n // val_freq)].append(xt_1.cpu().view(batch_size,-1))
      #     #     middle_states_true[str(n // val_freq)].append(xt_1_true.cpu().view(batch_size,-1))
      
      #     if n < middle_states_num:
      #         middle_states[str(n)].append(xt_1.cpu().view(batch_size,-1))
      #         middle_states_true[str(n)].append(xt_1_true.cpu().view(batch_size,-1))

      middle_states[str(0)].append(batch.cpu().view(batch_size,-1))
      # middle_states_true[str(0)].append(batch.cpu().view(batch_size,-1))

  middle_states_cat = {key: torch.cat(value,dim=0) for key, value in middle_states.items()}
  # middle_states_true_cat = {key: torch.cat(value,dim=0) for key, value in middle_states_true.items()}
  torch.save(middle_states_cat, os.path.join(save_path, "middle_states.pt"))
  # torch.save(middle_states_cat, os.path.join(save_path, "middle_states_true.pt"))

if __name__ == "__main__":
  app.run(main)