import numpy as np

def l1_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute L1 Loss (Mean Absolute Error).

    Parameters:
    y_true (np.ndarray): Ground truth values.
    y_pred (np.ndarray): Predicted values.

    Returns:
    float: L1 loss value.
    """
    return np.mean(np.abs(y_true - y_pred))


# Example
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.1, 7.8])

loss = l1_loss(y_true, y_pred)
print("L1 Loss:", loss)