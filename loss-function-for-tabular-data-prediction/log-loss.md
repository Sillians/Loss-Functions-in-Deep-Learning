# **Log Loss (Logarithmic Loss / Logistic Loss / Cross-Entropy Loss)**

### **Definition**

Log Loss is a performance metric (and loss function) for **classification problems**, especially probabilistic classifiers. It measures the **uncertainty of predictions** by comparing predicted probabilities to the actual class labels.

It penalizes **confident but wrong predictions** more heavily than less confident ones, making it a stricter evaluation metric than accuracy.



### **Mathematical Formula**

For **binary classification**:

$`L_{log} = -\frac{1}{N} \sum_{i=1}^{N} [ y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i) ]`$

For **multi-class classification**:

$`L_{log} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \cdot \log(\hat{y}_{i,c})`$

Where:

* $N$ = number of samples
* $C$ = number of classes
* $`y_{i,c}`$ = 1 if sample $i$ belongs to class $c$, else 0
* $`\hat{y}_{i,c}`$ = predicted probability of sample $i$ being in class $c$



### **Intuition**

* Log Loss rewards **accurate probability estimates** and penalizes overconfident wrong predictions.
* Example: Predicting 0.9 for the correct class is better than predicting 0.6, but predicting 0.9 for the wrong class is much worse than predicting 0.6 for the wrong class.
* Minimizing Log Loss pushes models to output probabilities that reflect the **true likelihood** of class membership.



### **Practical Use Cases**

* **Logistic Regression** → Uses Log Loss as its optimization objective.
* **Neural Networks (softmax layer)** → Trained with categorical cross-entropy (multi-class log loss).
* **Ensemble methods** → XGBoost, LightGBM, CatBoost often optimize Log Loss for classification tasks.
* **Evaluation metric** → Commonly used in machine learning competitions (e.g., Kaggle) to rank models based on probability calibration, not just accuracy.



**Key Insight**:

* Log Loss is stricter than accuracy: A model can have high accuracy but poor log loss if its predicted probabilities are not well-calibrated.

---
