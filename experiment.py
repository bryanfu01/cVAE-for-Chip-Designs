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
from core_engines.soft_drcs import SoftDRC


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict,
                 soft_drc_params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.soft_drc_params = soft_drc_params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        self.soft_drc_evaluator = SoftDRC(
            overlap_weight=self.soft_drc_params.get('overlap_weight', 50.0),
            area_weight=self.soft_drc_params.get('area_weight', 20.0),
            thermal_weight=self.soft_drc_params.get('thermal_weight', 10.0),
            sharpness_weight=self.soft_drc_params.get('sharpness_weight', 100.0),
            target_area=self.soft_drc_params.get('target_area', 20.0), # E.g., for a 10x10 footprint
            cohesion_weight=self.soft_drc_params.get('cohesion_weight', 20.0)
            )

    def forward(self, input: Tensor, condition: Tensor) -> Tensor:
        # Equivalent to self.model.forward(input), but pytorch works better this way for backprop
        return self.model(input, condition)

    def training_step(self, batch, batch_idx):
        layouts, heat_maps, _, powers = batch
        self.curr_device = layouts.device

        if batch_idx == 0 and self.current_epoch == 0:
            print("\n=== PROBE 1: DATA PIPELINE ===")
            print(f"Heatmap Min/Max: {heat_maps.min().item():.4f} / {heat_maps.max().item():.4f}")
            print(f"Layouts Shape:   {layouts.shape}")
            print(f"Sample Powers:   {powers[0].tolist()}")
            print("==============================\n")

        results = self.forward(input=layouts, condition=heat_maps)

        if batch_idx == 0:
            recons_probe = results[0]
            print(f"\n=== PROBE 2: EPOCH {self.current_epoch} RAW OUTPUT ===")
            print(f"Raw Prob Min/Max: {recons_probe.min().item():.4f} / {recons_probe.max().item():.4f}")
            
            # Check the average "mass" of the continuous macros. 
            # If target_area is 100, this should ideally climb toward 100 over time.
            mean_mass = recons_probe.sum(dim=(2, 3)).mean().item()
            print(f"Mean Macro Mass:  {mean_mass:.2f}")

        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'], 
                                              batch_idx=batch_idx)

        if self.soft_drc_params.get('use_soft_drc', False):
            recons = results[0]
            drc_metrics = self.soft_drc_evaluator(recons, heat_maps, powers)

            warmup_epochs = self.soft_drc_params.get('warmup_epochs', 30)
            warmup_factor = min(1.0, self.current_epoch / warmup_epochs)
            scaled_drc_loss = drc_metrics['total_drc_loss'] * warmup_factor + (1 - warmup_factor) * drc_metrics['Soft_Sharpness_Loss']

            # PROBE 3: Gradient Balance (Prints once per epoch)
            if batch_idx == 0:
                base_loss = train_loss['loss'].item()
                raw_drc = drc_metrics['total_drc_loss'].item()
                print(f"=== PROBE 3: EPOCH {self.current_epoch} LOSS BALANCE ===")
                print(f"Base VAE Loss:    {base_loss:.4f}")
                print(f"Raw Soft DRC:     {raw_drc:.4f}")
                print(f"Warmup Multiplier: {warmup_factor:.4f}")
                print(f"Effective DRC:    {(raw_drc * warmup_factor):.4f}\n")

            train_loss['loss'] = self.soft_drc_params.get('vanilla_weight') * train_loss['loss'] + scaled_drc_loss
            
            # Merge the isolated metrics for TensorBoard tracking
            train_loss.update({k: v for k, v in drc_metrics.items() if k != 'total_drc_loss'})

            train_loss['DRC_Warmup_Factor'] = torch.tensor(warmup_factor)
            train_loss.update({k: (v * warmup_factor) for k, v in drc_metrics.items() if k != 'total_drc_loss'})

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        layouts, heat_maps, _, powers = batch
        self.curr_device = layouts.device

        results = self.forward(layouts, condition=heat_maps)
        val_loss = self.model.loss_function(*results,
                                            M_N=1.0, 
                                            batch_idx=batch_idx)
        
        if self.soft_drc_params.get('use_soft_drc', False):
            recons = results[0]
            drc_metrics = self.soft_drc_evaluator(recons, heat_maps, powers)
            
            # Add the raw physics penalty directly to the total val_loss
            val_loss['loss'] = self.soft_drc_params.get('vanilla_weight', 100.0) * val_loss['loss'] + drc_metrics['total_drc_loss']
            
            # Merge the individual tracking metrics (Overlap, Area, Thermal)
            val_loss.update({k: v for k, v in drc_metrics.items() if k != 'total_drc_loss'})

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        
    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label, _, _ = next(iter(self.trainer.datamodule.test_dataloader()))
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
                                        condition = test_label)
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

