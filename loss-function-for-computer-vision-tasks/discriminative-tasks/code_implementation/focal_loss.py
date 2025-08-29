import numpy as np


class FocalLoss:
    def __init__(self, alpha = 0.25, gamma = 2.0, eps = 1e-7):
        """
        Initialize Focal Loss.
        Args:
            alpha: weight balancing factor for classes (default 0.25).
            gamma: focusing parameter to reduce loss for easy examples (default 2.0).
            eps: small value to avoid log(0).
        """
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, y_true, y_pred):
        """
        Compute Focal Loss.
        Args:
            y_true: Ground truth labels (0 or 1 for binary classification).
            y_pred: Predicted probabilities, same shape as y_true.
        Returns:
            Scalar focal loss value.
        """
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps) # numerical stability
        
        # For binary classification
        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        
        loss = - alpha_t * ((1 - p_t)**self.gamma) * np.log(p_t)
        return np.mean(loss)



print("\n=== IMBALANCED OBJECT DETECTION EXAMPLE ===")

# Simple Example 1:
y_true = np.array([1, 0, 1, 0])   # Ground truth
y_pred = np.array([0.9, 0.1, 0.2, 0.8])  # Predicted probabilities

loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
loss = loss_fn.forward(y_true, y_pred)
print("Focal Loss:", loss)        
       


# Example 2:
# Scenario: 90% background pixels, 10% object pixels (common in object detection)
np.random.seed(42)

# Create imbalanced dataset
y_imbalanced = np.array([1,1,1,0,0,0,0,0,0,0])  # 30% positive, 70% negative
y_pred_imbalanced = np.array([0.9, 0.3, 0.8, 0.05, 0.7, 0.1, 0.2, 0.01, 0.15, 0.08])
# Easy positive, hard positive, easy positive, easy negatives, hard negative, easy negatives...

# Compare Focal Loss with different parameters vs standard BCE
bce_loss = -np.mean(y_imbalanced * np.log(np.clip(y_pred_imbalanced, 1e-7, 1-1e-7)) + 
                   (1-y_imbalanced) * np.log(np.clip(1-y_pred_imbalanced, 1e-7, 1-1e-7)))

focal_gamma0 = FocalLoss(alpha=0.25, gamma=0.0)  # No focusing (like weighted BCE)
focal_gamma2 = FocalLoss(alpha=0.25, gamma=2.0)  # Standard focal loss
focal_gamma5 = FocalLoss(alpha=0.25, gamma=5.0)  # Strong focusing

print(f"Standard BCE Loss:     {bce_loss:.4f}")
print(f"Focal Loss (γ=0):      {focal_gamma0.forward(y_imbalanced, y_pred_imbalanced):.4f}")
print(f"Focal Loss (γ=2):      {focal_gamma2.forward(y_imbalanced, y_pred_imbalanced):.4f}")
print(f"Focal Loss (γ=5):      {focal_gamma5.forward(y_imbalanced, y_pred_imbalanced):.4f}")

print("\nKey Benefits:")
print("- Reduces loss for easy examples (confident predictions)")
print("- Focuses training on hard examples (low confidence)")
print("- Alpha balances positive/negative class importance")
print("- Higher γ = stronger focus on hard examples")


""" 
Scenario:
•  Imbalanced dataset: 30% positive (objects), 70% negative (background)
•  Mixed difficulty: Easy positives (0.9, 0.8), hard positive (0.3), hard negative (0.7)

Key Results:

Loss Comparison:
•  Standard BCE: 0.3372
•  Focal Loss (γ=0): 0.1763 (weighted BCE effect)
•  Focal Loss (γ=2): 0.0603 (standard focal loss)
•  Focal Loss (γ=5): 0.0202 (strong focusing)

Key Insights:

Progressive Reduction: Higher γ values progressively reduce the loss  
Easy Example Suppression: Confident predictions contribute less to loss  
Hard Example Focus: Low-confidence predictions get more attention  
Class Balance: Alpha parameter (0.25) weights positive class importance  


Why This Matters:

•  Object Detection: Most pixels are background (easy negatives)
•  Hard Mining: Focuses on challenging examples that need more learning
•  Training Efficiency: Prevents easy examples from dominating the gradient
"""