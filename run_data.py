import os
import yaml
import torch
from tqdm import tqdm
from data_generation.data_generator import DataGenerator

def main():
    # 1. Load the YAML Config
    config_path = os.path.join("configs", "data.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_cfg = config['data_params']
    solver_cfg = config['finite_solver_params']
    save_cfg = config['logging_params']
    # 2. Instantiate the Orchestrator
    generator = DataGenerator(data_cfg=data_cfg, solver_cfg=solver_cfg)
    
    layouts = []
    heat_maps = []
    
    # 3. Generate the Data
    print(f"Generating {data_cfg['num_samples']} synthetic chips...")
    for _ in tqdm(range(data_cfg['num_samples'])):
        layout, heatmap = generator.generate_valid_chip()
        
        layouts.append(layout)
        heat_maps.append(heatmap)
        
    # 4. Save to Drive
    save_dir = save_cfg['save_path']
    file_name = save_cfg['name'] + ".pt"
    full_path = os.path.join(save_dir, file_name)
    
    os.makedirs(save_dir, exist_ok=True) 
    
    torch.save({
        'layouts': torch.stack(layouts),
        'heatmaps': torch.stack(heat_maps)
    }, full_path)
    
    print(f"Dataset successfully saved to: {full_path}")
