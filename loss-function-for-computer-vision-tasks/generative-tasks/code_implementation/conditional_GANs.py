""" 
- In conditional GANs (cGANs), the generator is trained to generate data conditioned on some input (e.g., text, class label, or another image)
- the discriminator learns to distinguish between real and fake data given the same condition.
- D_loss trains the discriminator to separate real vs. fake.
- G_loss trains the generator to fool the discriminator.
"""


import torch
import torch.nn as nn

# Define cGAN Loss using Binary Cross-Entropy
class CGANLoss(nn.Module):
    def __init__(self):
        super(CGANLoss, self).__init__()
        self.loss = nn.BCELoss()
        
    def forward(self, D_real, D_fake):
        """
        Compute cGAN loss.
        
        Parameters:
        - D_real: Discriminator output for real pairs (x, y).
        - D_fake: Discriminator output for fake pairs (x, G(x, z)).

        Returns:
        - D_loss: Loss for discriminator.
        - G_loss: Loss for generator.
        """
        real_labels = torch.ones_like(D_real, device=D_real.device)
        fake_labels = torch.zeros_like(D_fake, device=D_fake.device)
        
        # Discriminator loss
        D_loss_real = self.loss(D_real, real_labels)
        D_loss_fake = self.loss(D_fake, fake_labels)
        D_loss = (D_loss_real + D_loss_fake) / 2
        
        # Generator loss (wants discriminator to think fake is real)
        G_loss = self.loss(D_fake, real_labels)
        
        return D_loss, G_loss
        

# Example
if __name__ == "__main__":
    # Simulated discriminator outputs
    D_real = torch.tensor([0.9, 0.95, 0.85], device="mps") # real pairs
    D_fake = torch.tensor([0.1, 0.2, 0.3], device="mps") # fake pairs
    
    cgan_loss = CGANLoss()
    D_loss, G_loss = cgan_loss(D_real, D_fake)
    
    print("Discriminator Loss:", D_loss.item())
    print("Generator Loss:", G_loss.item())
        
        
        