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
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

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
           ckpt_path="/content/drive/MyDrive/VAE_Checkpoints/last.ckpt")