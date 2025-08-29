## Action Recognition

Loss functions for action recognition and abnormal event detection in deep learning are crucial for training models that can accurately identify and interpret human activities or detect unusual events in video data.

For **action recognition**, where the task involves classifying sequences of frames into specific actions, the **Cross-Entropy Loss** is most commonly used. This loss compares the predicted action class probabilities with the ground-truth action labels, refining the model’s ability to distinguish between different actions over time. To account for the sequential nature of video data, **temporal loss functions** such as **Temporal Cross-Entropy Loss** are also employed. These ensure that models capture temporal dependencies effectively, improving recognition accuracy.

Recent advancements in action recognition have leveraged both `CNN-based` and `transformer-based` architectures, generally trained with `Cross-Entropy Loss`:

* **Non-local Networks** capture long-range temporal dependencies in video sequences.
* **Temporal Shift Modules (TSM)** efficiently model temporal dynamics with minimal computational overhead.
* **SlowFast Networks** utilize pathways operating at different temporal resolutions for improved performance.
* **Multiscale Vision Transformers (MViT)** adapt transformer-based models for multi-scale video data.
* **TimeSformer** is a transformer architecture tailored specifically for video understanding.
* **Video Swin Transformers** extend hierarchical Swin Transformers to the video domain.
* **Hierarchical Action Transformers (HAT)** capture relationships across multiple temporal scales.

In addition to Cross-Entropy Loss, **Contrastive Loss** has also been adopted in approaches such as action recognition models using contrastive learning. This loss ensures that similar action representations are drawn closer in the feature space, while dissimilar ones are pushed apart—enabling robust performance in zero-shot action recognition tasks.

---


### Description of Action Recognition Methods and Loss Functions

| Task               | Method        | Technique | Loss Function |
| ------------------ | ------------- | --------- | ------------- |
| Action Recognition | Wang et al.   | CNN       | Cross-Entropy |
|                    | TSM           | CNN       | Cross-Entropy |
|                    | SlowFast      | CNN       | Cross-Entropy |
|                    | MViT          | CNN, ViT  | Cross-Entropy |
|                    | TimeSformer   | ViT       | Cross-Entropy |
|                    | Video-Swin-Tr | ViT       | Cross-Entropy |
|                    | HAT           | ViT       | Cross-Entropy |
|                    | ActionCLIP    | CNN, ViT  | Contrastive   |

---


## **Common Loss Functions**


## **Cross-Entropy Loss**

**Definition**

Cross-Entropy Loss is one of the most widely used loss functions in classification tasks. It measures the dissimilarity between the predicted probability distribution of a model and the true distribution (ground truth labels). A perfect model would predict a probability of **1** for the correct class and **0** for all others. Cross-Entropy penalizes predictions that deviate from this ideal.



### **Mathematical Formulation**

For a single sample with true label $`y`$ and predicted probabilities $`\hat{y}`$:

* **Binary Cross-Entropy (BCE):**

$`L = - \left[ y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \right]`$

where:

* $`y \in \{0,1\}`$ (true label)
* $`\hat{y}`$ is the predicted probability of class 1



* **Categorical Cross-Entropy (CCE):**

$`L = - \sum_{i=1}^{C} y_i \cdot \log(\hat{y}_i)`$

where:

* $`C`$ is the number of classes
* $`y_i`$ is the true label (one-hot encoded, $`y_i=1`$ for the correct class, otherwise $`0`$)
* $`\hat{y}_i`$ is the predicted probability for class $`i`$



* **For a dataset with $`N`$ samples:**

$`\mathcal{L} = - \frac{1}{N} \sum_{j=1}^{N} \sum_{i=1}^{C} y_{ij} \cdot \log(\hat{y}_{ij})`$



### **Intuition**

* Cross-Entropy Loss evaluates **how confident the model is about the correct class**.
* If the model assigns high probability to the correct class, the loss is low.
* If the model assigns low probability, the loss increases dramatically (logarithmic penalty).
* This makes Cross-Entropy particularly effective for classification because it **encourages sharp, confident predictions** aligned with the true labels.



### **Practical Use Cases**

1. **Image Classification**

   * Used in CNNs and Vision Transformers for classifying images into categories (e.g., cats vs. dogs).

2. **Natural Language Processing (NLP)**

   * Applied in text classification, sentiment analysis, and machine translation tasks.

3. **Object Detection (classification part)**

   * Used to classify detected regions into object categories or background.

4. **Action Recognition**

   * Classifies sequences of video frames into specific actions.

5. **Speech Recognition**

   * Applied when mapping audio inputs to phoneme or word probability distributions.

---



## **Contrastive Loss**

**Definition**

Contrastive Loss is a metric learning loss function designed to **learn embeddings** such that similar samples are pulled closer together in the feature space, while dissimilar samples are pushed farther apart.
It is often used in tasks where the relationship between pairs of samples is more important than categorical classification, such as similarity learning or zero-shot recognition.



### **Mathematical Formulation**

For a pair of samples $`(x_1, x_2)`$ with label $`y`$:

* $`y = 1`$ if the pair is similar (positive pair)
* $`y = 0`$ if the pair is dissimilar (negative pair)

The **Contrastive Loss** is defined as:

$`L = y \cdot D^2 + (1 - y) \cdot \max(0, m - D)^2`$

where:

* $`D = \| f(x_1) - f(x_2) \|_2`$ is the Euclidean distance between embeddings of the two samples
* $`m`$ is a margin parameter ensuring dissimilar pairs are at least $`m`$ apart
* $`f(x)`$ is the embedding function (neural network mapping input to latent space)



### **Intuition**

* For **similar pairs** ($`y=1`$): The loss is $`D^2`$, so the model tries to **minimize the distance** between embeddings.
* For **dissimilar pairs** ($`y=0`$): The loss is $`\max(0, m-D)^2`$, meaning embeddings are only penalized if they are **closer than the margin** $`m`$.
* This creates a **structured embedding space** where semantically similar items cluster together, while different items are well separated.



### **Practical Use Cases**

1. **Face Verification (e.g., FaceNet)**

   * Determines whether two images belong to the same person by comparing embeddings.

2. **Zero-Shot Action Recognition (e.g., ActionCLIP)**

   * Learns alignment between actions and textual descriptions, enabling classification without explicit training labels.

3. **Siamese Networks**

   * Used in one-shot or few-shot learning tasks, where the model compares query samples to a small support set.

4. **Image Retrieval**

   * Ensures images of the same category are embedded close together, making retrieval efficient.

5. **Signature / Handwriting Verification**

   * Verifies whether two signatures or handwriting samples belong to the same individual.

---













































































