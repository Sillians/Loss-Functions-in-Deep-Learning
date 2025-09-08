# **Binary Cross-Entropy (BCE) Loss**

### **Definition**

Binary Cross-Entropy (BCE) is a loss function used for **binary classification tasks**. It measures the dissimilarity between the predicted probability $`\hat{y}`$ and the true label $`y \in {0,1}`$ for each sample (or pixel in segmentation). BCE penalizes incorrect predictions more when the model is confident but wrong.


### **Mathematical Formula**

For a single prediction:

$`L_{BCE} = -[ y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y})]`$

For $N$ samples:

$`L_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} [y_{i} \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i)]`$

Where:
* $y_i \in {0,1}$ = ground truth label
* $\hat{y}_i \in [0,1]$ = predicted probability


### **Intuition**

* BCE **rewards the model** when it assigns high probability to the correct class.
* If the true label is 1, the first term ($y \cdot \log(\hat{y})$) dominates, pushing $\hat{y}$ closer to 1.
* If the true label is 0, the second term ($(1-y) \cdot \log(1-\hat{y})$) dominates, pushing $\hat{y}$ closer to 0.
* Encourages **probabilistic confidence** rather than just discrete predictions.



### **Practical Use Cases**

* **Binary Classification** (spam vs. not spam, fraud detection, tumor vs. non-tumor).
* **Image Segmentation** (pixel-wise binary classification → foreground vs. background).
* **Autoencoders** (for binary data reconstruction tasks).
* **Multi-label classification** (where each label is treated as an independent binary decision).

---