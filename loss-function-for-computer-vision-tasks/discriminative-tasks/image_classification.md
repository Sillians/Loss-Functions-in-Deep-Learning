## **Image Classification**

In deep learning, loss functions play a central role in **image classification**, as they quantify how well a model’s predictions match the true labels. During training, these losses are used in **backpropagation** to compute gradients with respect to the model’s parameters, thereby guiding updates to minimize classification errors.

#### **Common Loss Functions**

* **Cross-Entropy Loss**

  * The most widely used for classification tasks.
  * Measures the performance of a model whose output is a probability distribution between `0` and `1`.
  * Encourages the network to assign `higher probabilities` to the `correct class`, improving performance in **multi-class settings**.

* **Mean Squared Error (MSE)**

  * Although primarily used for regression, MSE can also be applied in classification.
  * However, it is generally `less effective` than `cross-entropy` because it does not handle probability distributions as efficiently.

The choice of loss function directly impacts a model’s **ability to learn, generalize, and perform on unseen images**.

---

#### **Loss Functions in CNN-based Models**

Deep learning models for image classification often integrate different loss functions depending on their design:

* **VGG (Visual Geometry Group)** – Utilized **Euclidean loss** for training.
* **Inception** – Trained using **Cross-Entropy Loss**.
* **WideResNet** – Also employed **Cross-Entropy Loss**.

CNN-based architectures remain foundational backbones for various computer vision tasks.

---

#### **Loss Functions in Vision Transformers (ViTs)**

Vision Transformers (ViTs) also commonly rely on **Cross-Entropy Loss** for training. However, specialized tasks have introduced new loss formulations:

* **Private Inference ViTs** – Used **Cross-Entropy** with reduced-dimension attention and novel softmax approximations to improve efficiency.
  
* **Troj-ViT** – A ViT model designed for **backdoor attacks**, combining **Cross-Entropy Loss** and a custom **Attention-Target Loss** to manipulate outputs while preserving normal inference accuracy.
  
* **RepQ-ViT** – Addressed post-training quantization challenges by using **Ranking Loss** to preserve the relative order of attention scores before and after quantization, enhancing low-bit inference accuracy.

* **MGViT** – Applied **Cross-Entropy** and **Kullback-Leibler (KL) Divergence Loss** for optimization.
  
* **LFViT** – Leveraged **Cross-Entropy** and **KL Divergence Loss**, while reducing computational costs by processing low-resolution inputs first and focusing computation on selected regions, significantly improving throughput and FLOPs efficiency.

---

#### **Model–Loss Function Comparison**

| Task                 | Method            | Technique | Loss Function                    |
| -------------------- | ----------------- | --------- | -------------------------------- |
| Image Classification | VGG               | CNN       | Euclidean                        |
|                      | Inceptionv3       | CNN       | Cross-Entropy                    |
|                      | WideResNet        | CNN       | Cross-Entropy                    |
|                      | RNAViT            | CNN       | Cross-Entropy                    |
|                      | TrojViT           | ViT       | Cross-Entropy + Attention-Target |
|                      | RepQViT           | ViT       | Ranking                          |
|                      | MGViT             | ViT       | Cross-Entropy + Kullback-Leibler |
|                      | LFViT             | ViT       | Cross-Entropy + Kullback-Leibler |

---

#### **Key Insight**

While **Cross-Entropy Loss** remains the dominant choice for both `CNNs` and `ViTs` in classification, specialized architectures increasingly combine it with **auxiliary loss functions** (e.g., Attention-Target Loss, Ranking Loss, KL Divergence) to address **task-specific challenges** such as quantization, efficiency, and adversarial robustness.

---



