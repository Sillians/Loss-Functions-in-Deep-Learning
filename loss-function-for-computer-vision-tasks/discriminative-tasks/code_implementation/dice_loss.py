import numpy as np

def dice_loss(y_true, y_pred, eps=1e-12):
    """
    Dice Loss for binary segmentation.
    
    Parameters:
        y_true: Ground truth binary labels (0 or 1), shape (N,)
        y_pred: Predicted probabilities (0 to 1), shape (N,)
        eps: Small value to prevent division by zero
        
    Returns:
        Scalar Dice loss
    """
    # Flatten arrays to ensure compatibility
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Compute intersection and union
    intersection = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    
    # Compute Dice coefficient
    dice_coeff = (2 * intersection + eps) / (denominator + eps)
    
    # Dice loss
    return 1 - dice_coeff


# =============================================================================
# COMPREHENSIVE DICE LOSS EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=== DICE LOSS EXAMPLE USAGE ===")
    
    # Example 1: Basic 1D example
    print("\n1. Basic 1D Binary Segmentation Example:")
    y_true = np.array([1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8, 0.65, 0.2, 0.05, 0.7])
    
    loss_basic = dice_loss(y_true, y_pred)
    print(f"   Ground truth: {y_true}")
    print(f"   Predictions:  {y_pred}")
    print(f"   Dice Loss: {loss_basic:.4f}")
    print(f"   Dice Coefficient: {1 - loss_basic:.4f}")
    
    # Example 2: Perfect prediction (loss should be 0)
    print("\n2. Perfect Prediction Example:")
    y_true_perfect = np.array([1, 0, 1, 1, 0])
    y_pred_perfect = np.array([1.0, 0.0, 1.0, 1.0, 0.0])  # Perfect match
    
    loss_perfect = dice_loss(y_true_perfect, y_pred_perfect)
    print(f"   Ground truth: {y_true_perfect}")
    print(f"   Predictions:  {y_pred_perfect}")
    print(f"   Dice Loss: {loss_perfect:.6f} (should be ~0)")
    
    # Example 3: Worst prediction (loss should be 1)
    print("\n3. Worst Prediction Example:")
    y_true_worst = np.array([1, 1, 1, 1, 1])
    y_pred_worst = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Complete mismatch
    
    loss_worst = dice_loss(y_true_worst, y_pred_worst)
    print(f"   Ground truth: {y_true_worst}")
    print(f"   Predictions:  {y_pred_worst}")
    print(f"   Dice Loss: {loss_worst:.6f} (should be ~1)")
    
    # Example 4: 2D Image Segmentation (common use case)
    print("\n4. 2D Medical Image Segmentation Example:")
    # Simulate a small 8x8 medical image segmentation
    np.random.seed(42)
    
    # Create a synthetic ground truth mask (e.g., tumor segmentation)
    y_true_2d = np.zeros((8, 8))
    y_true_2d[2:6, 3:7] = 1  # Square region as "tumor"
    
    # Create predicted probabilities (slightly noisy)
    y_pred_2d = y_true_2d.copy().astype(float)
    y_pred_2d += np.random.normal(0, 0.1, y_pred_2d.shape)  # Add noise
    y_pred_2d = np.clip(y_pred_2d, 0, 1)  # Clip to valid probability range
    
    loss_2d = dice_loss(y_true_2d, y_pred_2d)
    
    print(f"   Image size: {y_true_2d.shape}")
    print(f"   True positives: {np.sum(y_true_2d)}")
    print(f"   Predicted sum: {np.sum(y_pred_2d):.2f}")
    print(f"   Dice Loss: {loss_2d:.4f}")
    print(f"   Dice Coefficient: {1 - loss_2d:.4f}")
    
    # Example 5: Comparing different prediction qualities
    print("\n5. Comparing Different Prediction Qualities:")
    
    # Ground truth
    y_true_comp = np.array([1, 1, 1, 0, 0, 0, 1, 1])
    
    # Good prediction
    y_pred_good = np.array([0.95, 0.9, 0.85, 0.1, 0.05, 0.15, 0.9, 0.88])
    
    # Mediocre prediction  
    y_pred_mediocre = np.array([0.7, 0.6, 0.65, 0.3, 0.4, 0.35, 0.75, 0.6])
    
    # Poor prediction
    y_pred_poor = np.array([0.6, 0.4, 0.5, 0.5, 0.6, 0.4, 0.3, 0.45])
    
    loss_good = dice_loss(y_true_comp, y_pred_good)
    loss_mediocre = dice_loss(y_true_comp, y_pred_mediocre)
    loss_poor = dice_loss(y_true_comp, y_pred_poor)
    
    print(f"   Ground truth: {y_true_comp}")
    print(f"   Good prediction - Loss: {loss_good:.4f}, Dice: {1-loss_good:.4f}")
    print(f"   Mediocre prediction - Loss: {loss_mediocre:.4f}, Dice: {1-loss_mediocre:.4f}")
    print(f"   Poor prediction - Loss: {loss_poor:.4f}, Dice: {1-loss_poor:.4f}")
    
    # Example 6: Handling edge cases
    print("\n6. Edge Cases:")
    
    # All zeros (background only)
    y_true_zeros = np.zeros(10)
    y_pred_zeros = np.random.uniform(0, 0.3, 10)  # Low predictions
    loss_zeros = dice_loss(y_true_zeros, y_pred_zeros)
    print(f"   All background case - Dice Loss: {loss_zeros:.4f}")
    
    # All ones (foreground only)
    y_true_ones = np.ones(10)
    y_pred_ones = np.random.uniform(0.7, 1.0, 10)  # High predictions
    loss_ones = dice_loss(y_true_ones, y_pred_ones)
    print(f"   All foreground case - Dice Loss: {loss_ones:.4f}")
    
    # Example 7: Batch processing simulation
    print("\n7. Batch Processing Example (Multiple Images):")
    
    batch_losses = []
    for i in range(3):
        # Simulate different images in a batch
        np.random.seed(i)
        y_true_batch = np.random.randint(0, 2, 20)
        y_pred_batch = np.random.uniform(0, 1, 20)
        
        # Apply some logic to make predictions more realistic
        y_pred_batch = np.where(y_true_batch == 1, 
                               y_pred_batch * 0.5 + 0.5,  # Higher probs for true positives
                               y_pred_batch * 0.5)        # Lower probs for true negatives
        
        loss_batch = dice_loss(y_true_batch, y_pred_batch)
        batch_losses.append(loss_batch)
        
        print(f"   Image {i+1}: Dice Loss = {loss_batch:.4f}, Dice Coeff = {1-loss_batch:.4f}")
    
    avg_batch_loss = np.mean(batch_losses)
    print(f"   Average Batch Loss: {avg_batch_loss:.4f}")
    
    print("\n=== DICE LOSS EXAMPLES COMPLETE ===")
    print("\nNote: Dice Loss is commonly used in:")
    print("- Medical image segmentation (organs, tumors, etc.)")
    print("- Semantic segmentation with imbalanced classes")
    print("- Binary segmentation tasks where overlap matters")
    print("- Cases where you want to penalize both false positives and false negatives")






""" 
Key Examples Shown:

1. Basic Usage: Simple 1D binary segmentation with predicted probabilities
•  Shows how dice loss decreases as predictions get better
•  Dice coefficient of 0.8243 indicates good overlap

2. Perfect vs Worst Cases: 
•  Perfect prediction: Dice Loss = 0 (complete overlap)
•  Worst prediction: Dice Loss = 1 (no overlap)

3. 2D Medical Image Segmentation: 
•  Simulates tumor segmentation in an 8x8 image
•  Shows how dice loss handles spatial data

4. Quality Comparison: 
•  Good predictions: Low dice loss (0.0838)
•  Poor predictions: High dice loss (0.4857)

5. Edge Cases: 
•  All background pixels
•  All foreground pixels

6. Batch Processing: Simulates processing multiple images


Key Properties of Dice Loss:

•  Range: [0, 1] where 0 is perfect, 1 is worst
•  Symmetric: Treats false positives and false negatives equally
•  Overlap-focused: Measures spatial overlap between prediction and ground truth
•  Class imbalance robust: Works well even when one class dominates

"""