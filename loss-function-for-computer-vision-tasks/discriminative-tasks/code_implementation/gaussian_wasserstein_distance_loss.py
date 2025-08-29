import numpy as np
from scipy.linalg import sqrtm


class GaussianWassersteinLoss:
    def __init__(self, eps=1e-7):
        """
        Initialize Gaussian Wasserstein Distance Loss.
        Args:
            eps: small value for numerical stability.
        """
        self.eps = eps
    
    def forward(self, mu_pred, sigma_pred, mu_true, sigma_true):
        """
        Compute Gaussian Wasserstein Distance Loss.
        Args:
            mu_pred: Predicted mean vector, shape (N, d).
            sigma_pred: Predicted covariance matrices, shape (N, d, d).
            mu_true: Ground truth mean vector, shape (N, d).
            sigma_true: Ground truth covariance matrices, shape (N, d, d).
        Returns:
            Scalar GW Distance loss value.
        """
        batch_size = mu_pred.shape[0]
        loss = 0.0
        
        for i in range(batch_size):
            mu_p = mu_pred[i]
            mu_t = mu_true[i]
            sigma_p = sigma_pred[i]
            sigma_t = sigma_true[i]
            
            # Mean distance term
            mean_diff = np.sum((mu_p - mu_t) ** 2)
            
            # Covariance distance term
            sqrt_sigma_t = sqrtm(sigma_t)
            inside_term = sqrt_sigma_t @ sigma_p @ sqrt_sigma_t
            cov_dist = np.trace(sigma_p + sigma_t - 2 * sqrtm(inside_term + self.eps * np.eye(sigma_p.shape[0])))
        
            # Wasserstein distance
            gw_dist = mean_diff + cov_dist
            loss += gw_dist
            
        return loss / batch_size
    

# Example usage
mu_true = np.array([[0, 0], [1, 1]])   # Ground truth means
mu_pred = np.array([[0.5, -0.2], [0.8, 1.2]])  # Predicted means

sigma_true = np.array([
    [[1, 0], [0, 1]],   # Identity covariance
    [[1.2, 0.3], [0.3, 1.5]]
])

sigma_pred = np.array([
    [[1.1, 0.1], [0.1, 0.9]],
    [[0.9, -0.2], [-0.2, 1.1]]
])

loss_fn = GaussianWassersteinLoss()
loss = loss_fn.forward(mu_pred, sigma_pred, mu_true, sigma_true)

print("Gaussian Wasserstein Distance Loss:", loss)





print("\n=== COMPREHENSIVE EXAMPLE: VARIATIONAL AUTOENCODER TRAINING ===")
print("Gaussian Wasserstein Loss for matching learned distributions to target distributions")

# Example: VAE latent space regularization
# Scenario: Training a VAE where the encoder outputs mean and covariance for latent distributions
# Goal: Match the learned latent distributions to target Gaussian priors

np.random.seed(42)
batch_size = 5
latent_dim = 3

# Target distributions (standard Gaussian priors)
target_means = np.zeros((batch_size, latent_dim))  # Zero means
target_covs = np.stack([np.eye(latent_dim) for _ in range(batch_size)])  # Unit covariances

# Predicted distributions from VAE encoder (with some deviations)
predicted_means = np.random.normal(0, 0.3, (batch_size, latent_dim))  # Slight mean deviations

# Predicted covariances (slightly different from identity)
predicted_covs = np.zeros((batch_size, latent_dim, latent_dim))
for i in range(batch_size):
    # Generate positive definite covariance matrices
    A = np.random.normal(0, 0.2, (latent_dim, latent_dim))
    predicted_covs[i] = np.eye(latent_dim) + 0.1 * (A @ A.T)  # Ensure positive definite

print(f"\nBatch size: {batch_size}, Latent dimension: {latent_dim}")
print(f"Target means (should be ~0): {target_means[0]}")
print(f"Target covariance (should be identity):\n{target_covs[0]}")
print(f"\nPredicted means: {predicted_means[0]}")
print(f"Predicted covariance:\n{predicted_covs[0]}")

# Compute Gaussian Wasserstein loss
loss_fn = GaussianWassersteinLoss(eps=1e-7)
gw_loss = loss_fn.forward(predicted_means, predicted_covs, target_means, target_covs)

print(f"\nGaussian Wasserstein Distance: {gw_loss:.6f}")

print("\nKey advantages of Gaussian Wasserstein Distance:")
print("  - More stable gradients compared to KL divergence")
print("  - Better handles cases where covariances become singular")
print("  - Provides meaningful distance even for non-overlapping distributions")
print("  - Symmetric distance measure (unlike KL divergence)")




"""
Scenario:
•  Training a VAE where the encoder outputs mean and covariance for latent distributions
•  Goal: Match learned latent distributions to standard Gaussian priors N(0,I)
•  Batch size: 5 samples, Latent dimension: 3

Key Components:

Target Distributions:
•  Mean: [0, 0, 0] (zero means)
•  Covariance: Identity matrix (unit covariances)

Predicted Distributions:
•  Slightly deviated means: [0.149, -0.041, 0.194]
•  Near-identity covariances with small perturbations

"""