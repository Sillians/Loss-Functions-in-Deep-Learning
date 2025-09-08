# **Cross-Entropy Loss**

### **Definition**

Cross-Entropy Loss is a widely used loss function in **classification tasks**. It measures the dissimilarity between the **true label distribution** $y$ and the **predicted probability distribution** $`\hat{y}`$. It builds on the concept of entropy, quantifying how uncertain predictions are relative to the truth.



### **Mathematical Formula**

For a single sample:

$`L_{CE} = - \sum_{c=1}^{C} y_c \cdot \log(\hat{y}_c)`$

For $N$ samples:

$`L_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \cdot \log(\hat{y}_{i,c})`$

* $`y_{i,c} \in {0,1}`$ → true label (often one-hot encoded).
* $`\hat{y}_{i,c} \in [0,1]`$ → predicted probability for class $c$.
* $`\sum_{c=1}^C \hat{y}_{i,c} = 1`$ (via softmax for multi-class).



### **Intuition**

* Cross-Entropy measures how "surprised" the model is by the true label.
* If the correct class is predicted with **high probability**, loss is small.
* If the model assigns **low probability** to the correct class, the loss is large.
* Encourages models to produce **confident and correct** predictions.



### **Practical Use Cases**

* **Binary classification** → Binary Cross-Entropy (e.g., spam detection).
* **Multi-class classification** → Categorical Cross-Entropy (e.g., ImageNet classification).
* **Multi-label classification** → BCE applied independently per label.
* Standard in **deep learning models** like CNNs, RNNs, and ViTs for classification tasks.


**Key Insight**:
Cross-Entropy is the **general formulation**, while **Binary Cross-Entropy** and **Categorical Cross-Entropy** are its **specialized cases**.

---



