import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import zipfile
from torch.utils.data import random_split


# Add your custom dataset class here
class ChipDataset(Dataset):
    def __init__(self, pt_file_path: str):
        # 1. Load the 50,000 sample dataset
        data = torch.load(pt_file_path)
        self.layouts = data['layouts']
        self.heatmaps = data['heatmaps']
        self.grid_size = 64

    def __len__(self):
        return self.layouts.shape[0]

    def __getitem__(self, idx):
        layout_coords = self.layouts[idx]
        heatmap = self.heatmaps[idx]
        
        # 2. Create a blank Cx64x64 continuous density grid
        num_macros = layout_coords.shape[1]
        rasterized_layout = torch.zeros((num_macros, self.grid_size, self.grid_size))
        
        # 3. Draw the macros onto the grid
        num_macros = layout_coords.shape[1]
        for i in range(num_macros):
            cx, cy, h, w = layout_coords[:, i].tolist()
            
            # Skip the -1 padding
            if cx == -1: 
                continue
                
            # Calculate grid boundaries
            x_start = max(0, int(round(cx - w/2)))
            x_end = min(self.grid_size, int(round(cx + w/2)))
            y_start = max(0, int(round(cy - h/2)))
            y_end = min(self.grid_size, int(round(cy + h/2)))
            
            # Map the rigid boundary to the continuous spatial density tensor
            rasterized_layout[i, y_start:y_end, x_start:x_end] = 1.0

        return rasterized_layout, heatmap 

class VAEDataset(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        full_dataset = ChipDataset(pt_file_path=self.data_dir)
        
        # 2. Calculate split sizes (e.g., 90% Train, 10% Val)
        total_len = len(full_dataset)
        train_len = int(0.9 * total_len)
        val_len = total_len - train_len
        
        # 3. Randomly split the dataset. 
        # Locking the manual_seed guarantees reproducibility for your final report metrics!
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, 
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42) 
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144, # Used for generating sample image grids
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     