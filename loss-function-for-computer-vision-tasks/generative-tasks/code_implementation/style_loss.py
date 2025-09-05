""" 
- Gram Matrix: Captures correlations between feature maps (texture & style).
- MSE Loss: Measures how close the style (Gram matrices) of generated and target images are.
- Multi-Layer Features: Usually extracted from several layers of a pre-trained VGG-19 for a richer style representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper function to compute Gram matrix
def gram_matrix(features):
    """
    Computes the Gram matrix of given feature maps.
    Input:
        features: (B, C, H, W) feature maps
    Returns:
        Gram matrix (B, C, C)
    """
    B, C, H, W = features.size()
    features = features.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))  # (B, C, C)
    return gram / (C * H * W)  # Normalize for stability


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, pred_features, target_features):
        """
        Computes Style Loss between predicted and target features.
        Inputs:
            pred_features: list of feature maps from generated image
            target_features: list of feature maps from target (style) image
        Returns:
            scalar style loss
        """
        style_loss = 0.0
        for pf, tf in zip(pred_features, target_features):
            gram_pred = gram_matrix(pf)
            gram_target = gram_matrix(tf)
            style_loss += F.mse_loss(gram_pred, gram_target)
        return style_loss



# Example Usage
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Fake feature maps (like from VGG layers)
    pred_features = [torch.rand(1, 64, 32, 32, device=device),
                     torch.rand(1, 128, 16, 16, device=device)]
    target_features = [torch.rand(1, 64, 32, 32, device=device),
                       torch.rand(1, 128, 16, 16, device=device)]

    criterion = StyleLoss().to(device)
    loss = criterion(pred_features, target_features)
    print(f"Style Loss: {loss.item():.4f}")




























