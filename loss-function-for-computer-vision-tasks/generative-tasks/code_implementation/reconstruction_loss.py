# Using Numpy
import numpy as np

def reconstruction_loss(x: np.ndarray, x_hat: np.ndarray, loss_type="mse") -> float:
    """
    Compute reconstruction loss between original input and reconstruction.

    Parameters:
        x (np.ndarray): Original input data
        x_hat (np.ndarray): Reconstructed data
        loss_type (str): "mse" or "l1"

    Returns:
        float: Reconstruction loss
    """
    if loss_type == "mse":
        # Mean Squared Error
        loss = np.mean((x - x_hat) ** 2)
    elif loss_type == "l1":
        # Mean Absolute Error
        loss = np.mean(np.abs(x - x_hat))
    else:
        raise ValueError("loss_type must be 'mse' or 'l1'")
    
    return loss


# Example 
x = np.array([1.0, 2.5, 3.6])
x_hat = np.array([0.9, 2.1, 3.2])

print("MSE Loss:", reconstruction_loss(x, x_hat, "mse"))
print("L1 Loss:", reconstruction_loss(x, x_hat, 'l1'))


# Using PyTorch
import torch

# Use MPS if available, else fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def reconstruction_loss_torch(x: torch.Tensor, x_hat: torch.Tensor, loss_type="mse") -> torch.Tensor:
    """
    Compute reconstruction loss using PyTorch (supports MPS).
    
    Parameters:
        x (torch.Tensor): Original input data
        x_hat (torch.Tensor): Reconstructed data
        loss_type (str): "mse" or "l1"

    Returns:
        torch.Tensor: Reconstruction loss
    """
    if loss_type == "mse":
        loss = torch.mean((x - x_hat) ** 2)
    elif loss_type == "l1":
        loss = torch.mean(torch.abs(x - x_hat))
    else:
        raise ValueError("loss_type must be 'mse' or 'l1'")
    
    return loss

x = torch.tensor([1.0, 2.5, 3.6], device=device)
x_hat = torch.tensor([0.9, 2.1, 3.2], device=device)

print("MSE Loss:", reconstruction_loss_torch(x, x_hat, "mse").item())
print("L1 Loss:", reconstruction_loss_torch(x, x_hat, "l1").item())