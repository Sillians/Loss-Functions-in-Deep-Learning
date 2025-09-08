# **Mean Absolute Error (MAE)**

### **Definition**

Mean Absolute Error (MAE) is a loss function commonly used in **regression tasks**. It measures the **average absolute difference** between predicted values and actual target values. Unlike MSE, it does not square the errors, which makes it **less sensitive to outliers**.



### **Mathematical Formula**

For $N$ samples:

$`MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|`$

Where:

* $`y_i`$ = actual (true) value
* $`\hat{y}_i`$ = predicted value
* $N$ = number of samples



### **Intuition**

* MAE evaluates how close predictions are to actual values by **directly averaging errors**.
* Each error contributes **linearly** (no squaring), so **outliers do not dominate** the loss.
* However, unlike MSE, MAE is **not differentiable at zero**, which can make optimization more challenging for gradient-based methods.
* A smaller MAE indicates better performance, representing the average deviation in the same units as the target variable.
* MAE provides a more **interpretable measure** of average error compared to MSE, especially when the scale of the target variable is important.
* MAE is often preferred when **all errors should be treated equally**, regardless of their magnitude.
* It is particularly useful in scenarios where **robustness to outliers** is desired, such as in financial data or real-world measurements.
* MAE can be more stable than MSE in the presence of noisy data, as it does not excessively penalize large deviations.
* However, the optimization landscape of MAE can be less smooth, potentially leading to slower convergence in some cases.
* In summary, MAE is a straightforward and robust loss function that provides an intuitive measure of prediction accuracy, making it a popular choice for many regression problems.
* When choosing between MAE and MSE, consider the nature of your data and the importance of outliers in your specific application.
* MAE is often used in conjunction with other metrics to provide a comprehensive evaluation of model performance.
* It is also commonly used in time-series forecasting, where stability and interpretability are crucial.
* Overall, MAE is a valuable tool in the machine learning practitioner's toolkit, especially when dealing with real-world data that may contain anomalies or outliers.
* Its simplicity and interpretability make it a go-to choice for many regression tasks across various domains.
* When implementing MAE in practice, ensure that your optimization algorithm can handle the non-differentiability at zero, or consider using smooth approximations if necessary.
* MAE can be combined with other loss functions in multi-objective optimization scenarios to balance different aspects of model performance.
* It is also worth noting that MAE can be sensitive to the scale of the target variable, so normalization or standardization may be beneficial in some cases.
* In conclusion, MAE is a robust and interpretable loss function that is well-suited for many regression tasks, particularly when outliers are present or when a straightforward measure of average error is desired.


### **Practical Use Cases**

* **Regression tasks with noisy data** where outliers should not heavily influence training (e.g., predicting house prices, income levels).
* **Robust evaluation metric** for model performance, since it provides an interpretable measure (average deviation in original units).
* **Time-series forecasting** where stability is important, and extreme values should not distort the loss too much.
  

---

**Key Insight**:
MAE is more **robust to outliers** than MSE but provides a **less smooth optimization landscape**. It is often chosen when fairness across all errors is more important than punishing large deviations.

---