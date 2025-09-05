""" 
- The KL Divergence between two probability distributions P(x) (true) and Q(x) (predicted) is:
"""

# Numpy Implementation
import numpy as np

def kl_divergence_numpy(p: np.ndarray, q: np.ndarray, eps=1e-10) -> float:
    """
    Compute KL Divergence between two probability distributions using NumPy.
    
    Parameters:
        p (np.ndarray): True distribution (must sum to 1)
        q (np.ndarray): Predicted distribution (must sum to 1)
        eps (float): Small constant to avoid log(0)
        
    Returns:
        float: KL divergence
    """
    p = np.clip(p, eps, 1)  # Avoid log(0)
    q = np.clip(q, eps, 1)
    
    return np.sum(p * np.log(p / q))

# Example
p = np.array([0.2, 0.5, 0.3])
q = np.array([0.1, 0.3, 0.6])

print("KL Divergence:", kl_divergence_numpy(p, q))



# PyTorch Implementation
import torch

def kl_divergence_pytorch(p: torch.Tensor, q: torch.Tensor, eps=1e-10) -> torch.Tensor:
    """
    Compute KL Divergence between two probability distributions using PyTorch.
    
    Parameters:
        p (torch.Tensor): True distribution (must sum to 1)
        q (torch.Tensor): Predicted distribution (must sum to 1)
        eps (float): Small constant to avoid log(0)
        
    Returns:
        torch.Tensor: KL divergence
    """
    p = torch.clamp(p, eps, 1)
    q = torch.clamp(q, eps, 1)
    
    return torch.sum(p * torch.log(p / q))

# Example 
p = torch.tensor([0.2, 0.5, 0.3])
q = torch.tensor([0.1, 0.3, 0.6])

print("KL Divergence (PyTorch):", kl_divergence_pytorch(p, q).item())