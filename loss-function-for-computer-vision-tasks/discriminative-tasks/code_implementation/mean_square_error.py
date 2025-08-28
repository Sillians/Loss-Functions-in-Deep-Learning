import numpy as np

class MSELoss:
    def __init__(self):
        self.y_true = None
        self.y_pred = None
    
    def forward(self, y_true, y_pred):
        """
        y_true: Ground truth values (numpy array).
        y_pred: Predicted values (numpy array).
        """
        self.y_true = y_true
        self.y_pred = y_pred
        return np.mean((y_true - y_pred) ** 2)
    
    
    def backward(self):
        """Compute the gradient of the MSE loss with respect to predictions."""
        n = self.y_true.shape[0]
        return (2 / n) * (self.y_pred - self.y_true)
    

# Example
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

mse = MSELoss()
loss = mse.forward(y_true, y_pred)
grad = mse.backward()

print("MSE Loss", loss)
print("Gradient", grad)
        
        
        
  
        