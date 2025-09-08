import numpy as np

def negative_log_likelihood(y_true, y_pred, epsilon=1e-15):
    """
    Compute the Negative Log-Likelihood (NLL) Loss.

    Parameters:
    y_true (np.ndarray): True class labels (one-hot encoded or class indices).
    y_pred (np.ndarray): Predicted probabilities for each class.
    epsilon (float): Small value to avoid log(0).

    Returns:
    float: NLL loss value.
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    
    # If y_true is one-hot encoded, convert to class indices
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # predicted probabilities for the true classes
    y_pred = y_pred[np.arange(len(y_true)), y_true]
    
    # log of predicted probabilities
    log_likelihood = np.log(y_pred)
    
    # Calculate NLL loss
    nll_loss = -np.mean(log_likelihood)
    return nll_loss

# Example usage
if __name__ == "__main__":
    # True labels (class indices)
    y_true = np.array([0, 2, 1, 2])
    
    # Predicted probabilities
    y_pred = np.array([[0.7, 0.2, 0.1], # correct class 0
                       [0.1, 0.3, 0.6], # correct class 2
                       [0.2, 0.5, 0.3], # correct class 1
                       [0.1, 0.4, 0.5]]) # correct class 2
    
    loss = negative_log_likelihood(y_true, y_pred)
    print(f"NLL Loss: {loss}")