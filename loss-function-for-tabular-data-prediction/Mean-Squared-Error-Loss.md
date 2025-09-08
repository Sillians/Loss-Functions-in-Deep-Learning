# **Mean Squared Error (MSE)**

### **Definition**

Mean Squared Error (MSE) is one of the most widely used loss functions for **regression problems**. It measures the **average squared difference** between predicted values and actual target values.

By squaring the errors, MSE penalizes **large deviations** more heavily than small ones, making it sensitive to outliers.



### **Mathematical Formula**

For $N$ samples:

$`MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2`$

Where:

* $`y_i`$ = actual (true) value
* $`\hat{y}_i`$ = predicted value
* $`N`$ = number of data points



### **Intuition**

* MSE evaluates how close predictions are to actual values.
* Squaring ensures all errors are **positive** and emphasizes **larger errors**.
* A smaller MSE indicates better performance.
* However, because of squaring, even a few large errors can dominate the loss, making the model sensitive to **outliers**.



### **Practical Use Cases**

* **Regression tasks** → Predicting continuous variables (e.g., house prices, stock prices, weather forecasts).
* **Optimization** → Used as an objective in linear regression and neural networks for regression.
* **Signal processing** → Measures reconstruction quality (e.g., denoising, compression).
* **Forecasting models** → Evaluates accuracy of time-series predictions.



**Key Insight**:
MSE provides a smooth, differentiable loss that works well for optimization, but its sensitivity to outliers sometimes makes **Mean Absolute Error (MAE)** or **Huber Loss** better alternatives.

---
