# **Categorical Cross-Entropy (CCE) Loss**

### **Definition**

Categorical Cross-Entropy (CCE) is a loss function used for **multi-class classification tasks**. It measures the dissimilarity between the predicted probability distribution $`\hat{y}`$ and the true class distribution $`y`$. The true label is typically one-hot encoded, meaning only one class is correct while all others are 0.



### **Mathematical Formula**

For a single sample with $C$ classes:

$`L_{CCE} = - \sum_{c=1}^{C} y_c \cdot \log(\hat{y}_c)`$

For $`N`$ samples:

$`L{CCE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \cdot \log(\hat{y}_{i,c})`$

* $`y_{i,c} \in {0,1}`$ = true label for class $`c`$ of sample $`i`$
* $`\hat{y}_{i,c} \in [0,1]`$ = predicted probability for class $`c`$ of sample $`i`$
* $`\sum_{c=1}^{C} \hat{y}_{i,c} = 1`$ (probability distribution after softmax)



### **Intuition**

* CCE **penalizes the model** when it assigns low probability to the true class.
* If the correct class has a high predicted probability, the loss is low.
* Forces the model to **distribute probabilities correctly** across multiple classes.
* Unlike Binary Cross-Entropy (BCE), CCE works with **mutually exclusive classes**.



### **Practical Use Cases**

* **Image Classification** (e.g., classifying an image as cat, dog, or horse).
* **Language Modeling** (predicting the next word from a vocabulary).
* **Speech Recognition** (selecting the correct phoneme or word class).
* **Vision Transformers (ViTs)** and **CNN-based models** for multi-class vision tasks.


**Key Difference from BCE**:

* **BCE** → Binary or multi-label (independent classes).
* **CCE** → Multi-class (mutually exclusive classes).

---
