import numpy as np

"""
1. Binary Cross-Entropy (BCE) Loss

Used for pixel-wise binary classification tasks (foreground vs. background).
"""

def binary_cross_entropy(y_true, y_pred, eps=1e-12):
    """
    Compute Binary Cross-Entropy Loss
    
    Parameters:
        - y_true: Ground truth binary labels (0 or 1), shape (N,)
        - y_pred: Predicted probabilities, shape (N,)
        - epsilon: Small value to prevent log(0)
        
    Returns:
        - Scalar BCE loss
    
    This function flattens the inputs and computes the mean BCE loss.
    For binary classification or binary segmentation tasks.
    """
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Compute BCE
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


# Example:
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.2, 0.8, 0.65, 0.1])
print("BCE Loss:", binary_cross_entropy(y_true, y_pred))







"""
2. Pixel-wise Cross-Entropy Loss
Definition (recap)

Used for multi-class segmentation tasks, where each pixel belongs to one of C classes.

"""


def pixel_wise_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Compute Pixel-wise Categorical Cross-Entropy loss for multi-class problems.
    
    Parameters:
    - y_true: One-hot encoded ground truth labels, shape (N, H, W, C) or similar.
    - y_pred: Predicted probabilities (after softmax), same shape as y_true.
    - epsilon: Small value to avoid log(0).
    
    Returns:
    - Scalar loss value.
    
    This assumes y_true is one-hot encoded along the last axis (classes).
    It computes cross-entropy per pixel and averages over all pixels and batch.
    Suitable for image segmentation tasks.
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Sum over classes, then mean over all other dimensions
    loss = -np.sum(y_true * np.log(y_pred), axis=-1)
    return np.mean(loss)




# Example:
y_true = np.array([
    [1, 0, 0],  # class 0
    [0, 1, 0],  # class 1
    [0, 0, 1]   # class 2
])

y_pred = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7]
])

print("Pixel-wise CE Loss:", pixel_wise_cross_entropy(y_true, y_pred))








