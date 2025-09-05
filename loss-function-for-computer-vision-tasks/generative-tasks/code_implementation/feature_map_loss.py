""" 
This loss directly compares the feature activations of a generated image and a target image from one or more layers of a pretrained network (e.g., VGG-19).

- Direct feature comparison: Uses MSE between feature maps from generated and target images.
- Content preservation: Ensures generated output retains semantic structure of the target.
Common in:
- Image inpainting (maintain structure while filling missing regions).
- Super-resolution (preserve high-level detail).
- Style transfer (used alongside Style Loss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureMapLoss(nn.Module):
    def __init__(self):
        super(FeatureMapLoss, self).__init__()

    def forward(self, pred_features, target_features):
        """
        Computes Feature Map Loss between predicted and target features.
        Inputs:
            pred_features: list of feature maps from generated image
            target_features: list of feature maps from target image
        Returns:
            scalar feature map loss
        """
        loss = 0.0
        for pf, tf in zip(pred_features, target_features):
            loss += F.mse_loss(pf, tf)  # Compare activations directly
        return loss



# Example
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Simulated feature maps (from different layers, e.g., VGG)
    pred_features = [torch.rand(1, 64, 32, 32, device=device),
                     torch.rand(1, 128, 16, 16, device=device)]
    target_features = [torch.rand(1, 64, 32, 32, device=device),
                       torch.rand(1, 128, 16, 16, device=device)]

    criterion = FeatureMapLoss().to(device)
    loss = criterion(pred_features, target_features)
    print(f"Feature Map Loss: {loss.item():.4f}")















