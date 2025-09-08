""" 
Intuition:
- In diffusion models, data is gradually corrupted by Gaussian noise.
- The model learns to predict the noise at each timestep.
- The Diffusion Loss (MSE) enforces the model to approximate this noise accurately.
"""

import numpy as np

def diffusion_loss_numpy(predicted_noise, true_noise):
    """
    Compute diffusion loss (MSE between predicted and true noise).
    
    Args:
        predicted_noise (ndarray): Model predicted noise (batch, dim).
        true_noise (ndarray): True Gaussian noise added (batch, dim).
    
    Returns:
        float: Diffusion loss value.
    """
    loss = np.mean((predicted_noise - true_noise) ** 2)
    return loss

# Example
pred_noise = np.array([[0.1, -0.2], [0.3, 0.5]])
true_noise = np.array([[0.0, -0.1], [0.4, 0.6]])

print("Diffusion Loss (NumPy):", diffusion_loss_numpy(pred_noise, true_noise))


import torch
import torch.nn as nn

def diffusion_loss_torch(predicted_noise, true_noise):
    """
    Compute diffusion loss using PyTorch (MSE Loss).
    
    Args:
        predicted_noise (Tensor): Model predicted noise (batch, dim).
        true_noise (Tensor): True Gaussian noise added (batch, dim).
    
    Returns:
        Tensor: Diffusion loss value.
    """
    criterion = nn.MSELoss()
    return criterion(predicted_noise, true_noise)

# Example
pred_noise = torch.tensor([[0.1, -0.2], [0.3, 0.5]], dtype=torch.float32)
true_noise = torch.tensor([[0.0, -0.1], [0.4, 0.6]], dtype=torch.float32)

print("Diffusion Loss (PyTorch):", diffusion_loss_torch(pred_noise, true_noise).item())
