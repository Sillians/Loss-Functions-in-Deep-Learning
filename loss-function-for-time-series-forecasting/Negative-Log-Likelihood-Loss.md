# **Negative Log-Likelihood (NLL) Loss**

**Definition**
Negative Log-Likelihood (NLL) Loss is a probabilistic loss function used in classification and probabilistic forecasting tasks. It measures how well a predicted probability distribution matches the actual target distribution by penalizing incorrect predictions with higher log penalties.


**Mathematical Formula**

For a single data point:

$`L = - \log \big( p(y \mid x; \theta) \big)`$

For $`N`$ samples:

$`L = - \frac{1}{N} \sum_{i=1}^{N} \log \big( p(y_i \mid x_i; \theta) \big)`$

Where:

* $`p(y_i \mid x_i; \theta)`$ = predicted probability assigned to the true class $`y_i`$
* $`\theta`$ = model parameters



**Intuition**

* If the model assigns a **high probability** to the correct class, the log term is close to zero → low loss.
* If the model assigns a **low probability** to the correct class, the log term becomes large (negative log) → high loss.
* Encourages the model to output calibrated probabilities, not just correct class labels.



**Practical Use Cases**

* **Classification tasks** (e.g., image recognition, text classification, speech recognition).
* **Probabilistic forecasting** in **time series** (e.g., **DeepAR** uses NLL to output probability distributions over future values).
* **Language modeling** and **NLP tasks**, where predicting likelihood of the next token is key.
* **Medical/finance applications**, where uncertainty quantification is critical.

---

