import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, condition: Tensor) -> Tensor:
        # Equivalent to self.model.forward(input), but pytorch works better this way for backprop
        return self.model(input, condition)

    def training_step(self, batch, batch_idx):
        layouts, heat_maps = batch
        self.curr_device = layouts.device

        results = self.forward(input=layouts, condition=heat_maps)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'], 
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        layouts, heat_maps = batch
        self.curr_device = layouts.device

        results = self.forward(layouts, condition=heat_maps)
        val_loss = self.model.loss_function(*results,
                                            M_N=1.0, 
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        
    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, condition = test_label)
        recons_vis = recons.data.sum(dim=1, keepdim=True)
        vutils.save_image(recons_vis,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            batch_size = test_input.size(0)
            samples = self.model.sample(num_samples=batch_size,
                                        current_device=self.curr_device,
                                        conditions = test_label)
            samples_vis = samples.cpu().data.sum(dim=1, keepdim=True)
            vutils.save_image(samples_vis,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        
        return optimizer

