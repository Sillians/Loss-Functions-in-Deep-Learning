import numpy as np
import math


def softmax(logits, axis=-1):
    z = logits - logits.max(axis=axis, keepdims=True)  # shift for stability
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=axis, keepdims=True)


def cross_entropy_with_logits(logits, targets, reduction="mean"):
    """
    Multiclass CE with integer class targets or one-hot targets.
    logits: [N, K]
    targets: [N] (int classes) or [N, K] (one-hot)
    """
    N = logits.shape[0]
    z = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True))
    log_probs = z - logsumexp  # [N, K]
    
    if targets.ndim == 1: # class indices
        nll = -log_probs[np.arange(N), targets]
    else:
        nll = -(targets * log_probs).sum(axis=1)
        
    
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    else:
        return nll # per-sample



def cross_entropy_grad_logits(logits, targets, reduction="mean"):
    """
    Gradient w.r.t. logits for softmax+CE.
    """
    N = logits.shape[0]
    probs = softmax(logits, axis=1)
    if targets.ndim == 1: # integer labels
        probs[np.arange(N), targets] -= 1.0
    else:
        probs -= targets
    
    if reduction == "mean":
        probs /= N
    return probs # [N, K]



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
    


def bce_with_logits(logits, targets, reduction="mean"):
    """
    Binary cross-entropy (with logits), stable.
    logits, targets: same shape
    """
    # Stable formula: max(z,0) - z*y + log(1 + exp(-|z|))
    z = logits
    loss = np.max(z, 0) - z * targets + np.log(1 + np.exp(-np.abs(z)))
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
    


def bce_with_logits_grad(logits, targets, reduction="mean"):
    grad = sigmoid(logits) - targets
    if reduction == "mean":
        return grad / logits.size
    elif reduction == "sum":
        return grad
    else:
        return grad # elementwise

    

    
# Multiclass Classification Examples
print("=== MULTICLASS CROSS-ENTROPY VALIDATION ===")

# Example 1 : Basic multiclass with integer targets
np.random.seed(0)
logits_basic = np.array([[2.0, 0.5, -1.0],
                        [0.1, 0.2, 0.3],
                        [-0.5, 1.8, 0.2]])  # [N=3, K=3]
targets_idx = np.array([0, 2, 1])  # true classes

loss_basic = cross_entropy_with_logits(logits_basic, targets_idx, reduction="mean")
grad_basic = cross_entropy_grad_logits(logits_basic, targets_idx, reduction="mean")
print("\n1. Basic multiclass (integer targets):")
print(f"   Logits shape: {logits_basic.shape}")
print(f"   Targets: {targets_idx}")
print(f"   CE loss: {float(loss_basic):.4f}")
print(f"   Grad shape: {grad_basic.shape}")
print(f"   Grad:\n{grad_basic}")


# Example 2 : One-hot encoded targets
np.random.seed(42)
logits_onehot = np.array([[1.2, -0.8, 2.1],
                         [-1.5, 0.9, 0.3],
                         [0.7, 1.1, -0.6],
                         [2.3, 0.1, -1.2]])  # [N=4, K=3]
targets_onehot = np.array([[0.0, 0.0, 1.0],  # class 2
                          [1.0, 0.0, 0.0],   # class 0
                          [0.0, 1.0, 0.0],   # class 1
                          [0.0, 0.0, 1.0]])  # class 2

loss_onehot = cross_entropy_with_logits(logits_onehot, targets_onehot, reduction="mean")
grad_onehot = cross_entropy_grad_logits(logits_onehot, targets_onehot, reduction="mean")
print("\n2. One-hot encoded targets:")
print(f"   Logits shape: {logits_onehot.shape}")
print(f"   One-hot targets shape: {targets_onehot.shape}")
print(f"   CE loss: {float(loss_onehot):.4f}")
print(f"   Grad shape: {grad_onehot.shape}")
print(f"   Grad:\n{grad_onehot}")



# Binary classification toy example for BCE validation
np.random.seed(42)
bce_logits = np.array([[1.5, -0.8],
                      [0.3, 2.1],
                      [-1.2, 0.7]])  # [N=3, 2] - binary logits
bce_targets = np.array([[1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0]])   # [N=3, 2] - binary targets (0 or 1)

bce_loss = bce_with_logits(bce_logits, bce_targets, reduction="mean")
bce_grad = bce_with_logits_grad(bce_logits, bce_targets, reduction="mean")
print("\nBCE with logits validation:")
print("BCE loss:", float(bce_loss))
print("BCE grad shape:", bce_grad.shape)
print("BCE grad:\n", bce_grad)

# Single output binary classification example
single_logits = np.array([2.0, -1.5, 0.8])  # [N=3] - single binary output
single_targets = np.array([1.0, 0.0, 1.0])  # [N=3] - binary targets

single_bce_loss = bce_with_logits(single_logits, single_targets, reduction="mean")
single_bce_grad = bce_with_logits_grad(single_logits, single_targets, reduction="mean")
print("\nSingle output BCE validation:")
print("Single BCE loss:", float(single_bce_loss))
print("Single BCE grad:", single_bce_grad)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



