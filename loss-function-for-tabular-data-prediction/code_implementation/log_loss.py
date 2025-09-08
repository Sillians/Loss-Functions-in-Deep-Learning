import numpy as np

def log_loss(y_true: np.array, y_pred: np.array) -> float:
    """
    Compute the Log Loss (Cross-Entropy Loss) for binary and multi-class classification.

    Parameters:
    y_true (np.ndarray): True labels, shape (N,) for binary or (N, C) for multi-class.
    y_pred (np.ndarray): Predicted probabilities, shape (N,) for binary or (N, C) for multi-class.

    Returns:
    float: Computed Log Loss.
    """
    # Ensure y_pred is within (0, 1) to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    if len(y_true.shape) == 1 or y_true.shape[1] == 1:
        # Binary classification
        y_true = y_true.flatten()
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        # Multi-class classification
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    return loss

# Example usage:
if __name__ == "__main__":
    # Binary classification example
    y_true_binary = np.array([1, 0, 1, 1, 0])
    y_pred_binary = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
    print("Binary Log Loss:", log_loss(y_true_binary, y_pred_binary))

    # Multi-class classification example
    y_true_multi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred_multi = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    print("Multi-class Log Loss:", log_loss(y_true_multi, y_pred_multi))