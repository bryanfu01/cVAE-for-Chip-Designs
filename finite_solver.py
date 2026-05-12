import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor

class FiniteDifferenceSolver:
    def __init__(self, grid_shape: tuple, k_conductivity: float, dx: float, iterations: int = 500):
        self.H, self.W = grid_shape
        self.k = k_conductivity
        self.dx = dx
        self.iterations = iterations
        
        self.K_avg = torch.tensor([[[[0.0, 0.25, 0.0],
                                     [0.25, 0.0, 0.25],
                                     [0.0, 0.25, 0.0]]]])

        self.K_avg = self.K_avg.to('cuda')

    def generate_power_map(self, macros: Tensor, power_vals: list) -> Tensor:
        """Projects discrete macro bounding boxes onto the continuous power grid."""

        device = macros.device
        P = torch.zeros((1, 1, self.H, self.W), device=device)
        
        num_macros = macros.shape[1]
        for i in range(num_macros):
            cx, cy, h, w = macros[:, i]
            
            x_start = max(0, int(round((cx - w/2).item())))
            x_end = min(self.W, int(round((cx + w/2).item())))
            y_start = max(0, int(round((cy - h/2).item())))
            y_end = min(self.H, int(round((cy + h/2).item())))
            
            area = (x_end - x_start) * (y_end - y_start)
            density = power_vals[i] / area if area > 0 else 0
            
            P[0, 0, y_start:y_end, x_start:x_end] += density
            
        return P

    def solve_heat_equation(self, P: Tensor) -> Tensor:
        """Diffuses the power map into a steady-state thermal gradient."""
        P_scaled = P * (self.dx**2 / (4 * self.k))
        T = torch.zeros_like(P)
        
        for _ in range(self.iterations):
            T_padded = F.pad(T, (1, 1, 1, 1), mode='constant', value=0.0)
            T = F.conv2d(T_padded, self.K_avg.to(T.device)) + P_scaled
            
        return T

    def simulate(self, macros: Tensor, power_vals: list) -> Tensor:
        """The main wrapper function to be called by the orchestrator."""
        P = self.generate_power_map(macros, power_vals)
        T_sim = self.solve_heat_equation(P)
        return T_sim