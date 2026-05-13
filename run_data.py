import os
import yaml
import torch
import concurrent.futures
from tqdm import tqdm
from data_generation.data_generator import DataGenerator 

# Create a global generator so the worker threads can access it
generator = None

def init_worker(config):
    """Initializes the DataGenerator once per CPU core."""
    global generator
    generator = DataGenerator(data_cfg=config['data_params'], solver_cfg=config['finite_solver_params'])

def generate_single_chip(_):
    """The function executed by each parallel thread."""
    return generator.generate_valid_chip()

def main():
    config_path = os.path.join("configs", "data.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    num_samples = config['data_params']['num_samples']
    
    print(f"Generating {num_samples} synthetic chips across multiple CPU cores...")
    
    available_cores = os.cpu_count()
    print(f"Detected {available_cores} CPU cores. Launching workers...")
    # 1. Launch a Process Pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=available_cores, initializer=init_worker, initargs=(config,)) as executor:
        
        # 2. Map the generation function across all cores and wrap it in tqdm for a progress bar
        results = list(tqdm(executor.map(generate_single_chip, range(num_samples)), total=num_samples))
        
    # 3. Unpack the results
    layouts, heat_maps = zip(*results)
        
    # 4. Save to Drive
    save_dir = config['logging_params']['save_path']
    file_name = config['logging_params']['name'] + ".pt"
    full_path = os.path.join(save_dir, file_name)
    
    os.makedirs(save_dir, exist_ok=True) 
    torch.save({
        'layouts': torch.stack(layouts),
        'heatmaps': torch.stack(heat_maps)
    }, full_path)
    
    print(f"Dataset successfully saved to: {full_path}")

if __name__ == "__main__":
    main()