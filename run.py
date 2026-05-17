import os
import yaml
import argparse
from pathlib import Path
import torch.backends.cudnn as cudnn

# Modern PyTorch Lightning Imports
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Local Project Imports
from models import *
from experiment import VAEXperiment
from dataset import VAEDataset

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()

# 1. Load the VAE config
with open(args.filename, 'r') as file:
    config = yaml.safe_load(file)

# 2. Load the Data config to sync parameters dynamically
data_config_path = os.path.join("configs", "data.yaml")
with open(data_config_path, "r") as f:
    data_config = yaml.safe_load(f)

# 3. Calculate dynamic channel sizes
# num_macros is a list like [10, 15], so index 1 is the max!
max_macros = data_config['data_params']['num_macros'][1] 

# Encoder: max_macros (for layout channels) + 1 (for the heat map condition)
config['model_params']['in_channels'] = max_macros + 1

# Decoder: needs to output exactly the number of macros
config['model_params']['out_channels'] = max_macros

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'])

# Modern seed_everything syntax
seed_everything(config['exp_params']['manual_seed'], workers=True)

# Build Model and Experiment
model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model, config['exp_params'])

# Safely check for 'accelerator' instead of the deprecated 'gpus' key
use_gpu = config['trainer_params'].get('accelerator') == 'gpu'
data = VAEDataset(**config["data_params"], pin_memory=use_gpu)

data.setup()

# Trainer initialized without the deprecated DDPPlugin
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath=os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor="val_loss",
                                     save_last=True),
                 ],
                 **config['trainer_params'])

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

print(f"======= Training {config['model_params']['name']} =======")

# Start training, resuming safely from your Google Drive checkpoint
runner.fit(experiment, 
           datamodule=data, 
           ckpt_path="/content/drive/MyDrive/ECE_175B_Final_Project/Vanilla_CVAE_Checkpoints/last.ckpt")