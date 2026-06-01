import os
import shutil
import yaml
import argparse
from pathlib import Path
import torch.backends.cudnn as cudnn

# Modern PyTorch Lightning Imports
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import ModelCheckpoint

# Local Project Imports
from models import *
from experiment import VAEXperiment
from dataset import VAEDataset
from callback import DriveSyncCallback

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/cvae.yaml')

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
experiment = VAEXperiment(model, config['exp_params'], config['soft_drc_params'])

# Safely check for 'accelerator' instead of the deprecated 'gpus' key
use_gpu = config['trainer_params'].get('accelerator') == 'gpu'
data = VAEDataset(**config["data_params"], pin_memory=use_gpu)

data.setup()

resume_path = config['trainer_params'].pop('resume_ckpt_path', None)

local_save_path = config['logging_params']['local_save_dir']
drive_save_path = config['logging_params']['drive_save_dir']

# 1. Standard checkpointing (Saves extremely fast to local Colab disk)
checkpoint_callback = ModelCheckpoint(
    dirpath=local_save_path,
    save_top_k=1,
    monitor="val_loss",
    save_last=True, # You can turn this back to True now!
)

# 2. Our custom safe-sync to Google Drive (e.g., every 5 epochs)
drive_sync = DriveSyncCallback(
    local_dir=local_save_path,
    drive_dir=drive_save_path,
    sync_every_n_epochs=10  # Adjust based on how fast your epochs run
)


runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     checkpoint_callback,
                     drive_sync
                 ],
                 **config['trainer_params'])

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

print(f"======= Training {config['model_params']['name']} =======")

# Start training, resuming safely from your Google Drive checkpoint

# Start training, resuming safely from your Google Drive checkpoint

golden_path = "/content/drive/MyDrive/ECE_175B_Final_Project/golden_vanilla_weights.ckpt"

if resume_path == golden_path:
    print(f"Loading GOLDEN WEIGHTS from: {resume_path}")
    print("Initiating Stage 2 Physics Fine-Tuning (Starting at Epoch 0)...")
    # weights_only=True forces the Trainer to drop the old Adam optimizer and epoch counter
    for param in experiment.model.encoder.parameters():
        param.requires_grad = False
    for param in experiment.model.fc_mu.parameters():
        param.requires_grad = False
    for param in experiment.model.fc_logvar.parameters():
        param.requires_grad = False
        
    print("Encoder frozen. Forcing Decoder to maintain latent diversity.")
    
    runner.fit(
        experiment, 
        datamodule=data, 
        ckpt_path=resume_path, 
        weights_only=True
    )
    
elif resume_path:
    print(f"Resuming standard training from: {resume_path}")
    # No weights_only flag: This fully restores the optimizer momentum and epoch counter!
    runner.fit(
        experiment, 
        datamodule=data, 
        ckpt_path=resume_path
    )
    
else:
    print("No resume path provided. Starting fresh training...")
    runner.fit(
        experiment, 
        datamodule=data
    )