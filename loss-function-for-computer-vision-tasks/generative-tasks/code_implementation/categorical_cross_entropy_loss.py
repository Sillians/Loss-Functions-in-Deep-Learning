import numpy as np

def categorical_cross_entropy(y_true, y_pred, eps=1e-12) -> float:
    """
    Categorical Cross-Entropy Loss (from scratch, NumPy).

    Args:
        y_true: one-hot encoded true labels, shape (N, C)
        y_pred: predicted probabilities (softmax outputs), shape (N, C)
        eps: small value to avoid log(0)

    Returns:
        loss (float)
    """
    # clip to prevent log(0)
    y_pred = np.clip(y_pred, eps, 1. - eps)
    
    # compute categorical cross-entropy
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss


# Example usage:
y_true = np.array([[0, 1, 0], [1, 0, 0]])  # 2 samples, class indices 1 and 0
y_pred = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])  # predicted probabilities

print("CCE Loss (NumPy):", categorical_cross_entropy(y_true, y_pred))




import torch
import torch.nn.functional as F

def categorical_cross_entropy(y_true, y_pred):
    """
    Categorical Cross-Entropy Loss (from scratch, PyTorch).
    
    Args:
        y_true: class indices, shape (N,)   (not one-hot, just integer labels)
        y_pred: logits, shape (N, C)        (raw outputs before softmax)

    Returns:
        loss (tensor)
    """
    # apply log softmax and gather correct class log probs
    log_probs = F.log_softmax(y_pred, dim=1)
    loss = -log_probs[range(y_true.shape[0]), y_true].mean()
    return loss


# Example usage (on MPS device):
device = torch.device("mps")

y_true = torch.tensor([1, 0], device=device)  # true labels
y_pred = torch.tensor([[0.1, 0.8, 0.1],
                       [0.7, 0.2, 0.1]], device=device)  # logits (not probs)

loss = categorical_cross_entropy(y_true, y_pred)
print("CCE Loss (PyTorch):", loss.item())
