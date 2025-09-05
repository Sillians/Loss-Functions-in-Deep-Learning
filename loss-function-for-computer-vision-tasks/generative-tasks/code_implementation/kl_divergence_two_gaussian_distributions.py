""" 
- True Distribution: P(x)
- Prior Distribution: Q(z)
"""

# Numpy Implementation
import numpy as np

def kl_divergence_gaussian_numpy(mu: np.ndarray, logvar: np.ndarray) -> float:
    """
    KL Divergence between N(mu, sigma^2) and N(0,1) using NumPy.
    
    Parameters:
        mu (np.ndarray): Mean of distribution
        logvar (np.ndarray): Log-variance of distribution (log(sigma^2))
    
    Returns:
        float: KL divergence
    """
    return 0.5 * np.sum(np.exp(logvar) + mu**2 - 1.0 - logvar)


# Example
mu = np.array([0.0, 0.5, -0.3])
logvar = np.array([0.0, -0.2, 0.1])  # log(sigma^2)

print("Gaussian KL Divergence (NumPy):", kl_divergence_gaussian_numpy(mu, logvar))



# PyTorch Implementation
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def kl_divergence_gaussian_torch(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL Divergence between N(mu, sigma^2) and N(0,1) using PyTorch.
    
    Parameters:
        mu (torch.Tensor): Mean of distribution
        logvar (torch.Tensor): Log-variance of distribution (log(sigma^2))
    
    Returns:
        torch.Tensor: KL divergence
    """
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar)


# Example
mu = torch.tensor([0.0, 0.5, -0.3], device=device)
logvar = torch.tensor([0.0, -0.2, 0.1], device=device)

print("Gaussian KL Divergence (PyTorch):", kl_divergence_gaussian_torch(mu, logvar).item())







