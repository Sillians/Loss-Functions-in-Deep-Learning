import numpy as np

def categorical_cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """
    Compute the Categorical Cross-Entropy Loss for multi-class classification.

    Parameters:
    y_true (array-like): True class labels (one-hot encoded).
    y_pred (array-like): Predicted probabilities for each class (softmax output).
    epsilon (float): Small value to avoid log(0).

    Returns:
    float: The average categorical cross-entropy loss for the batch.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute categorical cross-entropy loss for each sample
    losses = -np.sum(y_true * np.log(y_pred), axis=1)
    
    # Return average loss
    return np.mean(losses)

# Example usage:
if __name__ == "__main__":
    # True labels (one-hot encoded)
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    # Predicted probabilities (softmax output)
    y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    
    loss = categorical_cross_entropy_loss(y_true, y_pred)
    print(f"Categorical Cross-Entropy Loss: {loss}")