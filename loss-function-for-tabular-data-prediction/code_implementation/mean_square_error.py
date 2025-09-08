import numpy as np

def mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    y_true (array-like): Actual (true) values.
    y_pred (array-like): Predicted values.

    Returns:
    float: Mean Squared Error value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

# Example usage:
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print("Mean Squared Error:", mean_squared_error(y_true, y_pred))