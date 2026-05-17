import torch
import numpy as np
from torch import Tensor

class Legalizer():
    def __init__(self, grid_shape: tuple) -> None:
        """
        macros: Tensor of shape (4, C). 
        Row 0: X, Row 1: Y, Row 2: H, Row 3: W
        boundaries: (H, W)
        """
        self.H, self.W = grid_shape

    def make_legal(self, macros: Tensor) -> Tensor:
        areas = macros[2, :] * macros[3, :]
        sorted_indices = torch.argsort(areas, descending=True)
        self.sorted_macros = macros[:, sorted_indices]

        anchored_macros = []
        num_macros = self.sorted_macros.shape[1]
        
        for i in range(num_macros):
            current_mac = self.sorted_macros[:, i].tolist()

            legal_mac = self.spiral_search(anchored_macros, current_mac)
            anchored_macros.append(legal_mac)

        # Convert back to a (4, C) tensor to hand off to the physics solver
        return torch.tensor(anchored_macros).T

    def spiral_search(self, anchored_macros: list, mac: list) -> list:
        cx, cy, h, w = mac
        
        # The VAE outputs continuous floats. We must snap the starting center to the integer grid.
        cx, cy = int(round(cx)), int(round(cy))
        
        # Check radius 0 (the exact center predicted by the VAE)
        if self.is_valid(cx, cy, h, w, anchored_macros):
            return [cx, cy, h, w]
            
        # Maximum possible distance to check before giving up (e.g., the grid size)
        max_radius = max(self.H, self.W)
        
        for r in range(1, max_radius):
            # 1. Walk the TOP edge (Left to Right)
            for dx in range(-r, r + 1):
                if self.is_valid(cx + dx, cy + r, h, w, anchored_macros): return [cx + dx, cy + r, h, w]
                
            # 2. Walk the RIGHT edge (Top to Bottom)
            for dy in range(r - 1, -r - 1, -1):
                if self.is_valid(cx + r, cy + dy, h, w, anchored_macros): return [cx + r, cy + dy, h, w]
                
            # 3. Walk the BOTTOM edge (Right to Left)
            for dx in range(r - 1, -r - 1, -1):
                if self.is_valid(cx + dx, cy - r, h, w, anchored_macros): return [cx + dx, cy - r, h, w]
                
            # 4. Walk the LEFT edge (Bottom to Top)
            for dy in range(-r + 1, r):
                if self.is_valid(cx - r, cy + dy, h, w, anchored_macros): return [cx - r, cy + dy, h, w]
                
        raise ValueError("Grid is too congested. Could not find a valid location.")

    def is_valid(self, test_x, test_y, h, w, anchored_macros) -> bool:
        """
        Helper function to check both boundaries and overlaps simultaneously.
        """
        test_mac = [test_x, test_y, h, w]
        
        if self.detect_boundary(test_mac):
            return False
            
        for anchored in anchored_macros:
            if self.detect_overlap(test_mac, anchored):
                return False
                
        return True

    def detect_overlap(self, mac1, mac2) -> bool:
        mac1_x, mac1_y, mac1_h, mac1_w = mac1
        mac2_x, mac2_y, mac2_h, mac2_w = mac2
        
        x_overlap = min(mac1_x + mac1_w/2, mac2_x + mac2_w/2) - max(mac1_x - mac1_w/2, mac2_x - mac2_w/2)
        y_overlap = min(mac1_y + mac1_h/2, mac2_y + mac2_h/2) - max(mac1_y - mac1_h/2, mac2_y - mac2_h/2)
        
        return (x_overlap > 0) and (y_overlap > 0)
    
    def detect_boundary(self, mac) -> bool:
       mac_x, mac_y, mac_h, mac_w = mac
       left_edge = mac_x - mac_w/2
       right_edge = mac_x + mac_w/2
       bottom_edge = mac_y - mac_h/2
       upper_edge = mac_y + mac_h/2

       return left_edge < 0 or right_edge > self.W or bottom_edge < 0 or upper_edge > self.H