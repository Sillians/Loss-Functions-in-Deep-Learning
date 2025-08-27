import numpy as np



def softmax(logits, axis=-1):
    z = logits - logits.max(axis=axis, keepdims=True)  # shift for stability
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=axis, keepdims=True)



def kl_divergence_discrete(p, q, eps=1e-12, reduction="mean"):
    """
    D_KL(p || q) for discrete distributions.
    p, q: probabilities (sum to 1 along last axis)
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    kl = (p * (np.log(p) - np.log(q))).sum(axis=1)
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    else:
        return kl


def kl_divergence_grad_logits(p, logits_q, reduction="mean"):
    """
    Gradient of D_KL(p || softmax(logits_q)) w.r.t. logits_q.
    """
    q = softmax(logits_q, axis=1)
    grad = q - p
    if reduction == "mean":
        return grad / logits_q.shape[0]
    elif reduction == "sum":
        return grad
    else:
        return grad
    


print("=== KL DIVERGENCE EXAMPLES ===")

# Example 1: Basic KL with soft targets (teacher-student distillation)
np.random.seed(0)
logits_student = np.array([[2.0, 0.5, -1.0],
                          [0.1, 0.2, 0.3]])  # [N=2, K=3] - student predictions

# Teacher soft targets (more confident)
teacher_probs = np.array([[0.7, 0.2, 0.1],   # confident about class 0
                         [0.1, 0.2, 0.7]])   # confident about class 2

student_probs = softmax(logits_student, axis=1)
kl_basic = kl_divergence_discrete(teacher_probs, student_probs, reduction="mean")
grad_basic = kl_divergence_grad_logits(teacher_probs, logits_student, reduction="mean")

print("\n1. Teacher-Student Knowledge Distillation:")
print(f"   Teacher probs: \n{teacher_probs}")
print(f"   Student probs: \n{student_probs}")
print(f"   KL(teacher||student): {float(kl_basic):.4f}")
print(f"   Gradient shape: {grad_basic.shape}")
print(f"   Gradient: \n{grad_basic}")


# Example 2: Temperature scaling comparison
np.random.seed(42)
logits_raw = np.array([[3.0, 1.0, 0.5],
                      [1.5, 2.8, 0.2],
                      [0.8, 0.9, 2.1]])  # [N=3, K=3]

# Different temperature scales
temperatures = [1.0, 3.0, 5.0]  # 1.0 = no scaling, higher = softer
target_probs = np.array([[0.6, 0.3, 0.1],
                        [0.2, 0.7, 0.1],
                        [0.1, 0.2, 0.7]])

print("\n2. Temperature Scaling Effects:")
for temp in temperatures:
    # Apply temperature scaling
    scaled_logits = logits_raw / temp
    scaled_probs = softmax(scaled_logits, axis=1)
    kl_temp = kl_divergence_discrete(target_probs, scaled_probs, reduction="mean")
    
    print(f"   Temperature {temp}:")
    print(f"     Scaled probs: {scaled_probs[0]}  # first sample")
    print(f"     KL divergence: {float(kl_temp):.4f}")
    print(f"     Max probability: {np.max(scaled_probs, axis=1)}")



# Example 3: Asymmetry demonstration - KL(p||q) vs KL(q||p)
np.random.seed(123)
p_dist = np.array([[0.8, 0.15, 0.05],
                  [0.1, 0.8, 0.1]])
q_dist = np.array([[0.4, 0.4, 0.2],
                  [0.3, 0.4, 0.3]])

kl_pq = kl_divergence_discrete(p_dist, q_dist, reduction="mean")
kl_qp = kl_divergence_discrete(q_dist, p_dist, reduction="mean")

print("\n3. KL Divergence Asymmetry:")
print(f"   P distribution: {p_dist}")
print(f"   Q distribution: {q_dist}")
print(f"   KL(P||Q): {float(kl_pq):.4f}")
print(f"   KL(Q||P): {float(kl_qp):.4f}")
print(f"   Difference: {float(abs(kl_pq - kl_qp)):.4f}")
print("   Note: KL divergence is asymmetric - KL(P||Q) ≠ KL(Q||P)")




# Example 4: Edge cases - identical and very different distributions
identical_p = np.array([[0.33, 0.33, 0.34]])
identical_q = np.array([[0.33, 0.33, 0.34]])
very_different_p = np.array([[0.9, 0.05, 0.05]])
very_different_q = np.array([[0.1, 0.45, 0.45]])

kl_identical = kl_divergence_discrete(identical_p, identical_q, reduction="mean")
kl_different = kl_divergence_discrete(very_different_p, very_different_q, reduction="mean")

print("\n4. Edge Cases:")
print(f"   Identical distributions:")
print(f"     P = Q = {identical_p[0]}")
print(f"     KL(P||Q): {float(kl_identical):.6f} (should be ≈ 0)")
print(f"   Very different distributions:")
print(f"     P = {very_different_p[0]}")
print(f"     Q = {very_different_q[0]}")
print(f"     KL(P||Q): {float(kl_different):.4f} (high divergence)")




"""
Teacher-Student Knowledge Distillation
•  Demonstrates using KL divergence for model distillation
•  Shows how a student model learns from soft teacher targets
•  Includes gradient computation for training


2. Temperature Scaling Effects
•  Shows how temperature scaling affects probability distributions
•  Compares KL divergence at different temperature values (1.0, 3.0, 5.0)
•  Higher temperature = softer distributions = higher KL divergence from target


3. KL Divergence Asymmetry
•  Demonstrates that KL(P||Q) ≠ KL(Q||P)
•  Shows the mathematical asymmetry property
•  Important for understanding which direction to use in practice


4. Edge Cases Analysis
•  Identical distributions: KL divergence ≈ 0 (perfect match)
•  Very different distributions: High KL divergence (1.7578)
•  Helps understand the range and meaning of KL values
"""