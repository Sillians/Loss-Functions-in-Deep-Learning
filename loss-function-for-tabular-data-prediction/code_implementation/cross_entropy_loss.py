import numpy as np

def cross_entropy_loss_for_single_sample(y_true, y_pred, epsilon=1e-15):
    """
    Compute the Cross-Entropy Loss for a single sample.

    Parameters:
    y_true (int): The true class label (0 or 1 for binary classification).
    y_pred (float): The predicted probability for the positive class (between 0 and 1).
    epsilon (float): Small value to avoid log(0).

    Returns:
    float: The cross-entropy loss for the sample.
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute cross-entropy loss
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return loss

def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """
    Compute the Cross-Entropy Loss for a batch of samples.

    Parameters:
    y_true (array-like): True class labels (0 or 1 for binary classification).
    y_pred (array-like): Predicted probabilities for the positive class (between 0 and 1).
    epsilon (float): Small value to avoid log(0).

    Returns:
    float: The average cross-entropy loss for the batch.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute cross-entropy loss for each sample
    losses = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Return average loss
    return np.mean(losses)


def cross_entropy_loss_multiclass(y_true, y_pred, epsilon=1e-15):
    """
    Compute the Cross-Entropy Loss for multi-class classification.

    Parameters:
    y_true (array-like): True class labels (one-hot encoded).
    y_pred (array-like): Predicted probabilities for each class (softmax output).
    epsilon (float): Small value to avoid log(0).

    Returns:
    float: The average cross-entropy loss for the batch.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute cross-entropy loss for each sample
    losses = -np.sum(y_true * np.log(y_pred), axis=1)
    
    # Return average loss
    return np.mean(losses)