import numpy as np

def contrastive_loss_numpy(x1: np.ndarray, x2: np.ndarray, y: int, margin: float =1.0) -> float:
    """
    Contrastive Loss using NumPy.

    Parameters:
        x1 (np.ndarray): Embedding vector of sample 1
        x2 (np.ndarray): Embedding vector of sample 2
        y (int): 1 if similar, 0 if dissimilar
        margin (float): Margin for dissimilar pairs

    Returns:
        float: Contrastive loss
    """
    # Euclidean distance
    D = np.linalg.norm(x1 - x2)
    
    # Loss formula
    loss = y * (D**2) + (1 - y) * (np.maximum(0, margin - D)**2)
    return loss

# Example
x1 = np.array([0.2, 0.5, 0.1])
x2 = np.array([0.3, 0.45, 0.15])

print("Contrastive Loss (similar):", contrastive_loss_numpy(x1, x2, y=1))
print("Contrastive Loss (dissimilar):", contrastive_loss_numpy(x1, x2, y=0))
