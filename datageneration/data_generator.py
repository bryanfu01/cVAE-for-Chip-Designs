import torch
import numpy as np
import yaml
from legalizer import Discretizer
from finite_solver import FiniteDifferenceSolver
import random

class DataGenerator:
    def __init__(self, data_cfg: dict, solver_cfg: dict):
        """
        datacfg: data params.
        solvercfg: finite solver params
        """

        # 2. Unpack data generation parameters
        self.dataset_name = data_cfg['name']
        self.num_macros = data_cfg['num_macros']
        self.num_samples = data_cfg['num_samples']
        
        # Python lets you unpack lists of size 2 directly into two variables
        self.w_min, self.w_max = data_cfg['widths_range']
        self.h_min, self.h_max = data_cfg['heights_range']
        self.p_min, self.p_max = data_cfg['power_range']
        self.grid_h, self.grid_w = data_cfg['grid_size']

        # 3. Unpack finite solver parameters
        self.k = solver_cfg['conductivity']
        self.iterations = solver_cfg['iterations']
        
        # Safely evaluate the resolution if it was passed as a fraction string like '1/64'
        res_val = solver_cfg['resolution']
        self.dx = eval(res_val) if isinstance(res_val, str) else res_val

        self.legalizer = Discretizer(grid_shape=(self.grid_h, self.grid_w))
        self.finite_solver = FiniteDifferenceSolver(grid_shape=(self.grid_h,self.grid_w), 
                                                    k_conductivity=self.k,
                                                    dx=self.dx,
                                                    iterations=self.iterations)


    def generate_valid_chip(self):
        invalid_chip, power_vals = self.random_chips()
        valid_chip = self.legalizer.make_discrete(macros=invalid_chip)
        valid_heat_map = self.finite_solver.simulate(macros=valid_chip, power_vals=power_vals)

        return (valid_chip, valid_heat_map)

    def random_chips(self):
        macros = torch.zeros(4, self.num_macros)
        macros[0, :] = torch.randint(0, self.grid_w + 1, (1, self.num_macros))
        macros[1, :] = torch.randint(0, self.grid_h + 1, (1, self.num_macros))
        macros[2, :] = torch.randint(self.h_min, self.h_max + 1, (1, self.num_macros))
        macros[3, :] = torch.randint(self.w_min, self.w_max + 1, (1, self.num_macros))
        
        power_vals = [random.uniform(self.p_min, self.p_max) for _ in range(self.num_macros)]

        return macros, power_vals
        

    

