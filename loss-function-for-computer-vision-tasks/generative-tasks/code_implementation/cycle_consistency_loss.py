""" 
- We use L1 loss (nn.L1Loss) as the default metric because it preserves sharper details.
- The loss is computed as:
$||F(G(x)) - x||_1$ (cycle A → B → A)
$||G(F(y)) - y||_1$ (cycle B → A → B)
- The sum of both terms is returned as the final cycle-consistency loss.
"""

import torch
import torch.nn as nn

# Cycle Consistency Loss implementation
class CycleConsistencyLoss(nn.Module):
    def __init__(self, loss_fn=nn.L1Loss()):
        super(CycleConsistencyLoss, self).__init__()
        self.loss_fn = loss_fn  # Typically L1 loss is used
        
    def forward(self, real_A, cycled_A, real_B, cycled_B):
        """
        real_A: original images from domain A
        cycled_A: images reconstructed after A -> B -> A
        real_B: original images from domain B
        cycled_B: images reconstructed after B -> A -> B
        """
        loss_A = self.loss_fn(cycled_A, real_A) # A -> B -> A cycle
        loss_B = self.loss_fn(cycled_B, real_B) # B -> A -> B cycle
        return loss_A + loss_B

# Example usage
device = torch.device("mps")

# Dummy image tensors (batch_size=2, channels=3, height=64, width=64)
real_A = torch.rand(2, 3, 64, 64, device=device)
real_B = torch.rand(2, 3, 64, 64, device=device)

# Simulated cycled reconstructions
cycled_A = real_A + 0.05 * torch.randn_like(real_A, device=device)
cycled_B = real_B + 0.05 * torch.randn_like(real_B, device=device)

# Initialize loss
cycle_loss_fn = CycleConsistencyLoss()

# Compute cycle-consistency loss
loss = cycle_loss_fn(real_A, cycled_A, real_B, cycled_B)
print("Cycle-Consistency Loss:", loss.item())