import numpy as np

def l2_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute L2 Loss (Mean Squared Error).

    Parameters:
    y_true (np.ndarray): Ground truth values.
    y_pred (np.ndarray): Predicted values.

    Returns:
    float: L2 loss value.
    """
    return np.mean((y_true - y_pred)**2)


# Example
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.1, 7.9])

loss = l2_loss(y_true, y_pred)
print("L2 Loss:", loss)