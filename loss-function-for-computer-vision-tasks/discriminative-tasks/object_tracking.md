# **Object Tracking: Loss Functions**

Object tracking aims to **localize and follow moving objects** across consecutive video frames. Loss functions used in this task balance **spatial accuracy, identity preservation, and temporal stability**.

### **Core Loss Functions**

1. **Bounding Box Regression Loss (Smooth L1, IoU, Logistic Regression)**

   * Ensures accurate localization of objects in each frame.
   * Penalizes differences between predicted and ground-truth bounding box coordinates.
   * Examples: Smooth L1 loss stabilizes regression; IoU loss maximizes overlap with ground-truth.

2. **Re-Identification Loss (ReID Loss – Contrastive, Triplet)**

   * Preserves **object identity** across frames by learning embeddings.
   * Encourages embeddings of the same object to stay close while pushing different objects apart.

3. **Temporal Consistency Loss**

   * Promotes smooth trajectories over time.
   * Penalizes abrupt changes in bounding box position or size across consecutive frames.

---

## **Notable Deep Learning Methods for Object Tracking**

| **Method**         | **Architecture**              | **Loss Function(s)**                          |
| ------------------ | ----------------------------- | --------------------------------------------- |
| **SiamFC**         | Siamese CNN                   | Cross-Correlation Loss                        |
| **Deep SORT**      | CNN + Kalman Filter           | Mahalanobis Distance                          |
| **Track R-CNN**    | CNN (extension of Mask R-CNN) | Softmax Cross-Entropy, Smooth L1              |
| **ATOM**           | CNN                           | IoU Loss                                      |
| **DiMP**           | CNN                           | Logistic Regression Loss                      |
| **OSTrack**        | Transformer-based             | Multi-task Loss (classification + regression) |
| **TransTrack**     | Transformer-based             | Focal Loss, IoU Loss, L1 Loss                 |
| **FairMOT**        | CNN (multi-object)            | Re-Identification Loss                        |
| **Diffusion MOT**  | Diffusion + Detection         | Focal Loss, L1 Loss, Generalized IoU (GIoU)   |
| **ReID-based MOT** | Detection + Re-ID             | L1 Loss, Re-Identification Loss               |

---


## **Object Tracking: Methods and Loss Functions**

| **Method**         | **Architecture**    | **Loss Function(s)**              |
| ------------------ | ------------------- | --------------------------------- |
| **STTA**           | CNN                 | Cross-Entropy, Triplet            |
| **SiamFC**         | ANN                 | Cross-Correlation                 |
| **Deep SORT**      | Kalman Filter + CNN | Mahalanobis Distance              |
| **Track R-CNN**    | CNN                 | Softmax Cross-Entropy, Smooth L1  |
| **ATOM**           | CNN, ResNet         | IoU                               |
| **DiMP**           | CNN, ResNet         | Logistic Regression               |
| **OSTrack**        | CNN                 | IoU, L1                           |
| **TransTrack**     | ViT                 | Focal, IoU, L1                    |
| **FairMOT**        | CNN                 | Re-Identification                 |
| **DiffusionTrack** | Diffusion           | Focal, Generalized IoU (GIoU), L1 |
| **MotionTrack**    | ViT                 | L1                                |

---

## **Logistic Regression Loss**

**Definition**

Logistic Regression Loss (also known as **Log Loss** or **Binary Cross-Entropy Loss**) is used when the task involves **binary classification** — i.e., predicting whether a sample belongs to a class (1) or not (0).

In **object tracking**, logistic regression loss is often used in discriminative trackers (e.g., **DiMP**) to decide whether a candidate region in the current frame corresponds to the target object or not.



### **Mathematical Formula**

For a single prediction:

$`L = -[y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y})]`$

For multiple predictions (N samples):

$`L = -\frac{1}{N} \sum_{i=1}^{N} \big[y_i \cdot \log(\hat{y}_i) + (1-y_i) \cdot \log(1-\hat{y}_i)\big]`$

Where:

* $y\_i \in {0,1}$ → ground truth label (object present or not).
* $\hat{y}\_i \in [0,1]$ → predicted probability (output of the sigmoid function).



### **Intuition**

* Converts raw model outputs (logits) into probabilities using the **sigmoid function**:

  $`\hat{y} = \frac{1}{1 + e^{-z}}`$

* Penalizes **wrong confident predictions** more heavily than uncertain predictions.

* If the true label is **1 (object present)**, the model is penalized when $\hat{y}$ is close to 0.

* If the true label is **0 (object absent)**, the model is penalized when $\hat{y}$ is close to 1.

This makes it suitable for **object/background discrimination** in tracking tasks.



### **Practical Use Cases in Tracking**

* **DiMP (Discriminative Model Prediction):**
  Uses logistic regression loss to train its classification branch. This enables the tracker to adaptively classify regions as either belonging to the tracked object or background.

* **Object Detection → Tracking Pipelines:**
  Logistic loss is commonly used in detection heads that feed into tracking frameworks (since detecting the object correctly is the first step).

* **Binary Foreground-Background Segmentation for Tracking:**
  Used when separating moving objects from the static background in tracking-by-detection approaches.

---
