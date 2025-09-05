""" 
- This is the core loss used in diffusion models (e.g., Stable Diffusion).
- The goal: the model learns to predict the added noise at each timestep and is trained using MSE loss between true and predicted noise.


- The loss = MSE(ε, ε̂) where ε is the true Gaussian noise and ε̂ is the model’s predicted noise.
- x_t is constructed using the forward diffusion process.
- The model takes (x_t, t) as input and predicts the noise.
- This is exactly how Stable Diffusion, DDPMs, and guided diffusion are trained.
"""

import torch
import torch.nn as nn

class NoisePredictionLoss(nn.Module):
    def __init__(self):
        super(NoisePredictionLoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, model, x0, t):
        """ 
        Compute noise prediction loss for diffusion models.
        
        Args:
            model: the neural network predicting noise ε_θ(x_t, t)
            x0: original clean image (B, C, H, W)
            t: timesteps (B,) tensor with integers in [0, T)
        
        Returns:
            scalar noise prediction loss
        """
        # Sample random Gaussian noise
        noise = torch.randn_like(x0)
        
        # Example: beta schedule (linear variance schedule)
        T = 1000 # number of diffusion steps
        beta = torch.linspace(1e-4, 0.02, T, device=x0.device) # variance
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        
        # Get corresponding alpha_bar for given t
        alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)
        
        # forward diffusion: create noisy image x_t
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        # Predict noise with the model
        noise_pred = model(xt, t)
        
        # Compute MSE loss between true and predicted noise
        loss = self.mse(noise_pred, noise)
        return loss


# Dummy UNet-like model for predicting noise
class DummyModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DummyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )
        
    def forward(self, x, t):
        # ignore t for dummy model
        return self.net(x)
    


# Example
B, C, H, W = 4, 3, 64, 64
x0 = torch.randn(B, C, H, W) # clean images
t = torch.randint(0, 1000, (B,)) # random timesteps

model = DummyModel()
criterion = NoisePredictionLoss()

loss = criterion(model, x0, t)
print("Noise Prediction Loss:", loss.item())