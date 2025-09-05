import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

""" 
- VGG16 pretrained on ImageNet as the feature extractor
- intermediate VGG features (e.g., relu3_3 or relu4_3).
- The loss is MSE in feature space, not pixel space.
- You can combine multiple layers (e.g., ['relu2_2', 'relu3_3']) for richer perceptual features.
"""

class PerceptualLoss(nn.Module):
    def __init__(self, layers=['relu3_3'], device="mps"):
        super(PerceptualLoss, self).__init__()
        
        # load pretrained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Map human-readable layer names to indices
        self.layer_name_mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }
        
        # Extract chosen layers
        self.layers = layers
        self.vgg = vgg[:max([self.layer_name_mapping[l] for l in layers]) + 1]
        
        # Send to device (MPS by default)
        self.device = torch.device(device if torch.backends.mps.is_available else "cpu")
        self.vgg = self.vgg.to(self.device)


    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: Generated image (B, C, H, W)
            y: Ground truth image (B, C, H, W)
        Returns:
            Perceptual Loss (scalar)
        """
        loss = 0.0
        for name, module in self.vgg._modules.items():
            x = module(x)
            y = module(y)
            
            if int(name) in [self.layer_name_mapping[l] for l in self.layers]:
                loss += torch.nn.functional.mse_loss(x, y)
        return loss


# Usage Example with MPS
# Dummy example 
gen_img = torch.randn(1, 3, 224, 224)
real_img = torch.randn(1, 3, 224, 224)

loss_fn = PerceptualLoss(layers=['relu3_3'], device="mps")

gen_img = gen_img.to(loss_fn.device)
real_img = real_img.to(loss_fn.device)

loss = loss_fn(gen_img, real_img)
print("Perceptual Loss:", loss.item())