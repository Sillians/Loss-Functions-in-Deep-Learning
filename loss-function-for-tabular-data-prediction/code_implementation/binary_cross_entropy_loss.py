import numpy as np

def binary_cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """
    Compute the Binary Cross-Entropy Loss for binary classification.

    Parameters:
    y_true (array-like): True class labels (0 or 1).
    y_pred (array-like): Predicted probabilities for the positive class (between 0 and 1).
    epsilon (float): Small value to avoid log(0).

    Returns:
    float: The average binary cross-entropy loss for the batch.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute binary cross-entropy loss for each sample
    losses = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Return average loss
    return np.mean(losses)

# Example usage:
if __name__ == "__main__":
    # True labels
    y_true = np.array([1, 0, 1, 1, 0])
    
    # Predicted probabilities
    y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
    
    loss = binary_cross_entropy_loss(y_true, y_pred)
    print(f"Binary Cross-Entropy Loss: {loss}")