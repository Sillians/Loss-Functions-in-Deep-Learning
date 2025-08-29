import numpy as np


class SmoothL1Loss:
    def __init__(self, delta: float = 1.0):
        """
        Initialize Smooth L1 (Huber) Loss.
        Args:
            delta: threshold between L2 and L1 behavior (default=1.0).
        """
        self.delta = delta
        self.y_true = None
        self.y_pred = None
        
     
    def forward(self, y_true, y_pred):
        """
        Compute Smooth L1 (Huber) Loss.
        Args:
            y_true: Ground truth values (numpy array).
            y_pred: Predicted values (numpy array).
        Returns:
            Scalar loss value.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        diff = np.abs(y_true - y_pred)
        loss = np.where(diff < self.delta,
                        0.5 * diff ** 2,
                        self.delta * (diff - 0.5 * self.delta))
        return np.mean(loss)
    
    def backward(self):
        """
        Compute gradient of Smooth L1 Loss with respect to predictions.
        Returns:
            Gradient array.
        """
        diff = self.y_pred - self.y_true
        grad = np.where(np.abs(diff) < self.delta,
                        diff,
                        self.delta * np.sign(diff))
        return grad / self.y_true.shape[0]
    


# What happens in the class above:

"""
* **`forward()`**: Computes the Smooth L1 loss.

  * Quadratic penalty (like MSE) when error < δ.
  * Linear penalty (like MAE) when error ≥ δ.

* **`backward()`**: Returns the gradient wrt predictions.
"""



# Example Usage
# Example 1: Object Detection Bounding Box Regression
print("\n1. Object Detection Bounding Box Regression:")
print("   (Common use case in R-CNN, YOLO, etc.)")

# Simulated bounding box coordinates: [x_center, y_center, width, height]
# Ground truth bounding boxes (normalized coordinates)
gt_boxes = np.array([[0.5, 0.3, 0.2, 0.4],   # Object 1
                     [0.7, 0.6, 0.15, 0.25],  # Object 2
                     [0.2, 0.8, 0.3, 0.2]])   # Object 3

# Predicted bounding boxes (with some errors)
pred_boxes = np.array([[0.52, 0.28, 0.22, 0.38],  # Small errors
                       [0.75, 0.65, 0.12, 0.30],   # Moderate errors
                       [0.1, 0.9, 0.4, 0.15]])     # Larger errors

# Use different delta values for comparison
deltas = [0.1, 1.0, 2.0]

for delta in deltas:
    loss_fn = SmoothL1Loss(delta=delta)
    loss = loss_fn.forward(gt_boxes, pred_boxes)
    grad = loss_fn.backward()
    
    print(f"\n   Delta = {delta}:")
    print(f"     Total loss: {loss:.4f}")
    print(f"     Loss per box: {np.mean(np.sum(np.abs(gt_boxes - pred_boxes), axis=1)):.4f}")
    print(f"     Gradient norm: {np.linalg.norm(grad):.4f}")
    
    # Show individual coordinate errors
    coord_errors = np.abs(gt_boxes - pred_boxes)
    print(f"     Max coordinate error: {np.max(coord_errors):.4f}")
    print(f"     Coordinates with |error| > delta: {np.sum(coord_errors > delta)}")


# Visualize loss behavior vs L2 and L1 losses
print("\n   Loss Comparison (for largest error = 0.2):")
error_range = np.linspace(0, 0.3, 100)
delta = 1.0

l2_loss = 0.5 * error_range**2
l1_loss = error_range
smooth_l1 = np.where(error_range < delta,
                     0.5 * error_range**2,
                     delta * (error_range - 0.5 * delta))

print(f"     At error=0.05: L2={0.5*0.05**2:.4f}, L1={0.05:.4f}, SmoothL1={smooth_l1[int(0.05*100/0.3)]:.4f}")
print(f"     At error=0.15: L2={0.5*0.15**2:.4f}, L1={0.15:.4f}, SmoothL1={smooth_l1[int(0.15*100/0.3)]:.4f}")
print(f"     At error=0.25: L2={0.5*0.25**2:.4f}, L1={0.25:.4f}, SmoothL1={smooth_l1[int(0.25*100/0.3)]:.4f}")


"""
Example 1: Object Detection Bounding Box Regression 

This example shows how SmoothL1Loss is used in object detection models like R-CNN, YOLO, etc.:

•  Simulated bounding boxes: Ground truth vs predicted coordinates
•  Different delta values (0.1, 1.0, 2.0): Shows how the threshold affects behavior
•  Loss comparison: Compares SmoothL1 with pure L1 and L2 losses at different error levels

Key Results:
•  All delta values gave same loss (0.0018) since max error (0.1) was small
•  Shows smooth transition behavior at different error magnitudes
"""




# Example 2: Robust Regression with Outliers
print("\n\n2. Robust Regression with Outliers:")
print("   (Comparing robustness to outliers vs L2 loss)")

# Generate regression data with outliers
np.random.seed(42)
n_samples = 20

# Clean targets (linear relationship)
clean_targets = np.linspace(0, 10, n_samples)

# Predictions with some noise
predictions_clean = clean_targets + np.random.normal(0, 0.5, n_samples)

# Add outliers to some predictions
predictions_with_outliers = predictions_clean.copy()
outlier_indices = [3, 8, 15, 18]  # Add outliers at these positions
predictions_with_outliers[outlier_indices] += np.array([5, -6, 4, -5])  # Large errors

print(f"\n   Dataset: {n_samples} samples")
print(f"   Outliers at indices: {outlier_indices}")
print(f"   Outlier errors: {predictions_with_outliers[outlier_indices] - clean_targets[outlier_indices]}")

# Compare different loss functions
loss_functions = {
    'L2 (MSE)': lambda y_true, y_pred: np.mean((y_true - y_pred)**2),
    'L1 (MAE)': lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
    'SmoothL1 (δ=0.5)': lambda y_true, y_pred: SmoothL1Loss(0.5).forward(y_true, y_pred),
    'SmoothL1 (δ=1.0)': lambda y_true, y_pred: SmoothL1Loss(1.0).forward(y_true, y_pred),
    'SmoothL1 (δ=2.0)': lambda y_true, y_pred: SmoothL1Loss(2.0).forward(y_true, y_pred)
}

print("\n   Loss Comparison:")
print("   Dataset Type        |  L2 (MSE)  |  L1 (MAE)  | SmoothL1(0.5) | SmoothL1(1.0) | SmoothL1(2.0)")
print("   " + "-"*85)

# Clean data (no outliers)
losses_clean = []
for name, loss_fn in loss_functions.items():
    loss = loss_fn(clean_targets, predictions_clean)
    losses_clean.append(loss)
    
print(f"   Clean data          |  {losses_clean[0]:8.4f}  |  {losses_clean[1]:8.4f}  |   {losses_clean[2]:9.4f}  |   {losses_clean[3]:9.4f}  |   {losses_clean[4]:9.4f}")

# Data with outliers
losses_outliers = []
for name, loss_fn in loss_functions.items():
    loss = loss_fn(clean_targets, predictions_with_outliers)
    losses_outliers.append(loss)
    
print(f"   With outliers       |  {losses_outliers[0]:8.4f}  |  {losses_outliers[1]:8.4f}  |   {losses_outliers[2]:9.4f}  |   {losses_outliers[3]:9.4f}  |   {losses_outliers[4]:9.4f}")

# Robustness analysis
print("\n   Robustness Analysis:")
for i, name in enumerate(loss_functions.keys()):
    increase_ratio = losses_outliers[i] / losses_clean[i]
    print(f"     {name:15s}: {increase_ratio:.2f}x increase due to outliers")

print("\n   Key Insights:")
print("     - L2 loss is heavily affected by outliers (quadratic penalty)")
print("     - L1 loss is robust but non-differentiable at zero")
print("     - SmoothL1 combines benefits: robust to outliers + smooth gradients")
print("     - Smaller δ values are more robust to large errors")

# Gradient analysis
print("\n   Gradient Analysis (for sample with large error):")
large_error_sample = np.array([5.0])  # Target
outlier_prediction = np.array([10.0])  # Prediction (error = 5.0)

for delta in [0.5, 1.0, 2.0]:
    loss_fn = SmoothL1Loss(delta=delta)
    loss_fn.forward(large_error_sample, outlier_prediction)
    grad = loss_fn.backward()
    
    # Compare with L2 gradient
    l2_grad = 2 * (outlier_prediction - large_error_sample)
    
    print(f"     δ={delta}: SmoothL1 grad={grad[0]:+.2f}, L2 grad={l2_grad[0]:+.2f}")
    
    

"""
Example 2: Robust Regression with Outliers

This demonstrates the robustness advantage of SmoothL1Loss:

•  Clean data vs outliers: 20 samples with 4 large outlier errors
•  Comprehensive comparison: L2 (MSE), L1 (MAE), and SmoothL1 with different δ values
•  Robustness analysis: How much each loss increases due to outliers

Key Results:
•  L2 (MSE): 26.36x increase (very sensitive to outliers)
•  L1 (MAE): 3.48x increase (most robust)
•  SmoothL1 (δ=0.5): 5.84x increase (good balance)
•  SmoothL1 (δ=1.0): 9.27x increase
•  SmoothL1 (δ=2.0): 15.98x increase
"""