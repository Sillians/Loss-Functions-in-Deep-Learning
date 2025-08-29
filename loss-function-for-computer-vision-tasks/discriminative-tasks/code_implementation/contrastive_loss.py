import numpy as np

def contrastive_loss(y_true: np.array, x1: np.array, x2: np.array, margin=1.0) -> float:
    """
    Compute Contrastive Loss for a batch of pairs.

    Parameters:
    -----------
    y_true : np.array
        Binary labels (1 for similar, 0 for dissimilar) with shape (batch_size,).
    x1 : np.array
        Embeddings for first set of inputs with shape (batch_size, embedding_dim).
    x2 : np.array
        Embeddings for second set of inputs with shape (batch_size, embedding_dim).
    margin : float
        Margin that defines how far apart dissimilar pairs should be.

    Returns:
    --------
    loss : float
        Average contrastive loss over the batch.
    """
    
    # Euclidean distance between embeddings
    distances = np.linalg.norm(x1 - x2, axis=1)
    
    # Loss for similar pairs (y=1): D^2
    positive_loss = y_true * np.square(distances)
    
    # Loss for dissimilar pairs (y=0): max(0, margin - D)^2
    negative_loss = (1 - y_true) * np.square(np.maximum(0, margin - distances))
    
    # Average loss over batch
    loss = np.mean(positive_loss + negative_loss)
    return loss


# Example usage
if __name__ == "__main__":
    # Example embeddings (batch_size=3, embedding_dim=2)
    x1 = np.array([[1.0, 2.0], [0.5, 1.5], [3.0, 3.0]])
    x2 = np.array([[1.2, 2.1], [2.0, 2.5], [2.5, 2.0]])

    # Labels: 1 = similar, 0 = dissimilar
    y_true = np.array([1, 0, 1])

    loss_value = contrastive_loss(y_true, x1, x2, margin=1.0)
    print("Contrastive Loss:", loss_value)
    
    
    
    print("\n=== FACE VERIFICATION EXAMPLE ===")
    print("Training embeddings to distinguish same vs different people")
    
    # Simulate face embeddings from a neural network (128-dimensional)
    np.random.seed(42)
    embedding_dim = 128
    batch_size = 8
    
    # Generate realistic face embeddings
    # Same person pairs: embeddings should be close
    same_person_base = np.random.normal(0, 1, (batch_size//2, embedding_dim))
    same_person_x1 = same_person_base + np.random.normal(0, 0.1, same_person_base.shape)  # Small noise
    same_person_x2 = same_person_base + np.random.normal(0, 0.1, same_person_base.shape)  # Small noise
    
    # Different person pairs: embeddings should be far apart
    diff_person_x1 = np.random.normal(0, 1, (batch_size//2, embedding_dim))
    diff_person_x2 = np.random.normal(2, 1, (batch_size//2, embedding_dim))  # Different distribution
    
    
    # Combine embeddings and labels
    face_x1 = np.vstack([same_person_x1, diff_person_x1])
    face_x2 = np.vstack([same_person_x2, diff_person_x2])
    face_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # First 4 are same person, last 4 are different
    
    # Calculate distances for analysis
    distances = np.linalg.norm(face_x1 - face_x2, axis=1)
    
    print(f"\nBatch size: {batch_size}, Embedding dimension: {embedding_dim}")
    print(f"Same person distances: {distances[face_labels==1]}")
    print(f"Different person distances: {distances[face_labels==0]}")
    
     # Test different margin values
    margins = [0.5, 1.0, 2.0]
    print(f"\n{'Margin':<8} | {'Loss':<10} | {'Effect':<30}")
    print("-" * 52)
    
    for margin in margins:
        loss = contrastive_loss(face_labels, face_x1, face_x2, margin=margin)
        
        if margin == 0.5:
            effect = "Small margin - less separation"
        elif margin == 1.0:
            effect = "Standard margin"
        else:
            effect = "Large margin - more separation"
            
        print(f"{margin:<8} | {loss:<10.4f} | {effect:<30}")
    
    print("\n=== KEY INSIGHTS ===")
    print("• Similar pairs (y=1): Loss = distance²")
    print("• Dissimilar pairs (y=0): Loss = max(0, margin - distance)²")
    print("• Pulls similar pairs closer, pushes dissimilar pairs apart")
    print("• Margin controls minimum separation for dissimilar pairs")
    print("• Essential for face verification, image retrieval, similarity learning")
