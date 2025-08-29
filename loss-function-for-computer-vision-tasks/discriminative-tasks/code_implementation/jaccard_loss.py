import numpy as np

def jaccard_loss(y_true, y_pred, eps=1e-12):
    """
    Jaccard Loss (IoU Loss) for binary segmentation.
    
    Parameters:
        y_true: Ground truth binary labels (0 or 1), shape (N,)
        y_pred: Predicted probabilities (0 to 1), shape (N,)
        eps: Small constant to avoid division by zero.
        
    Returns:
        Scalar Jaccard loss
    """
    # Flatten for safety
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Intersection and union
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    
    # IoU
    iou = (intersection + eps) / (union + eps)
    
    # Jaccard loss
    return 1 - iou


# Example usage:
y_true = np.array([1, 0, 1, 1, 0, 0, 1])
y_pred = np.array([0.9, 0.2, 0.8, 0.65, 0.1, 0.05, 0.7])

print("Jaccard Loss (IoU Loss):", jaccard_loss(y_true, y_pred))