import numpy as np

def l2_loss(y_pred, y_true, reduction="mean"):
    """
    Euclidean (sum-of-squares) loss.
    y_pred, y_true: same shape
    reduction: "mean", "sum", or "none"    
    """
    
    diff = y_pred - y_true
    loss = 0.5 * np.square(diff) # 1/2 ||diff||^2 per element
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss # element wise
    


def l2_gradient(y_pred, y_true, reduction="mean"):
    diff = y_pred - y_true
    if reduction == "mean":
        return diff / y_pred.size
    elif reduction == "sum":
        return diff
    else:
        return diff # elementwise grad
    


y_pred = np.array(np.random.randn(1, 3))
y_true = np.array(np.random.randn(1, 3))
l2 = l2_loss(y_pred, y_true)
print(l2)


l2_sum = l2_loss(y_pred, y_true, "sum")
print(l2_sum)