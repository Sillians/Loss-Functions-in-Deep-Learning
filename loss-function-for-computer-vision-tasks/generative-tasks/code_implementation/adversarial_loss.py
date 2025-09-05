""" 
- This is the core objective of GANs, where the generator tries to fool the discriminator, and the discriminator tries to distinguish real from fake samples.
- Vanilla GAN → uses BCE loss.
- LSGAN → replaces BCE with MSE loss, more stable gradients.
- WGAN → uses mean difference instead of BCE/MSE, requires weight clipping or gradient penalty.
"""

import torch
import torch.nn as nn

class AdversarialLoss(nn.Module):
    def __init__(self, mode="vanilla"):
        """
        Args:
            mode: "vanilla" for BCE-based GAN loss,
                  "lsgan" for Least Squares GAN,
                  "wgan" for Wasserstein GAN
        """
        super(AdversarialLoss, self).__init__()
        self.mode = mode
        if mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":
            self.loss = nn.MSELoss()
            
    
    def forward(self, preds, target_is_real: bool):
        """
        Args:
            preds: discriminator predictions (batch,)
            target_is_real: bool → True for real, False for fake
        Returns:
            scalar adversarial loss
        """
        if self.mode == "wgan":
            return -preds.mean() if target_is_real else preds.mean()
        
        # for vanilla or LSGAN
        target = torch.ones_like(preds) if target_is_real else torch.zeros_like(preds)
        return self.loss(preds, target)


# Example using Dummy discriminator outputs
real_preds = torch.randn(4, 1) # D(x) for real samples
fake_preds = torch.randn(4, 1) # D(G(z)) for fake samples

criterion = AdversarialLoss(mode="vanilla")

# Discriminator loss
d_loss_real = criterion(real_preds, True)
d_loss_fake = criterion(fake_preds, False)
d_loss = (d_loss_real + d_loss_fake) / 2

# Generator loss (tries to fool D, so we label fake as real)
g_loss = criterion(fake_preds, True)

print("Discriminator Loss:", d_loss.item())
print("Generator Loss:", g_loss.item())