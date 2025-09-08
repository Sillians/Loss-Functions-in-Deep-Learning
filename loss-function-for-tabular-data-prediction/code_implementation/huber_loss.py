import numpy as np

def huber_loss(y_true: np.array, y_pred: np.array, delta=1.0):
    """
    Compute the Huber loss between true and predicted values.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.
    delta (float): The threshold at which to change between quadratic and linear loss.

    Returns:
    float: The computed Huber loss.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta

    squared_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)

    return np.where(is_small_error, squared_loss, linear_loss).mean()


# Example usage
if __name__ == "__main__":
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    loss = huber_loss(y_true, y_pred, delta=1.0)
    print(f"Huber Loss: {loss}")    