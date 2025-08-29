import numpy as np

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def logistic_regression_loss(y_true, y_pred_logits):
    """
    Logistic Regression Loss (Binary Cross-Entropy).
    
    Parameters:
    y_true : numpy array of shape (N,)
        Ground truth labels (0 or 1).
    y_pred_logits : numpy array of shape (N,)
        Raw model outputs (logits before sigmoid).
    
    Returns:
    float
        The average logistic regression loss.
    """
    # Apply sigmoid to convert logits into probabilities
    y_pred = sigmoid(y_pred_logits)
    
    # Add a small epsilon to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Binary cross-entropy formula
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss



# Ground truth labels
y_true = np.array([1, 0, 1, 0])

# Logits (raw outputs from a model)
y_pred_logits = np.array([2.0, -1.0, 1.5, -2.0])  # before sigmoid

loss_value = logistic_regression_loss(y_true, y_pred_logits)
print("Logistic Regression Loss:", loss_value)
