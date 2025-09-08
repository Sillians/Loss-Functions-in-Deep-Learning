
# **Gini Impurity Loss**

### **Definition**

Gini Impurity is a metric used in **decision trees** (e.g., CART algorithm) to measure the **degree of impurity or disorder** in a dataset. It represents the probability of incorrectly classifying a randomly chosen element if it were labeled according to the **class distribution** at a given node.

Unlike cross-entropy, which uses logarithms, Gini impurity is based on squared probabilities, making it computationally simpler.



### **Mathematical Formula**

For a node with $C$ classes:

$`Gini = 1 - \sum_{c=1}^{C} p_c^2`$

Where:

* $`p_c`$ = proportion of samples belonging to class $c$ in the node.

**Range**:

* $0$ → Perfectly pure node (all samples from one class).
* Maximum → When classes are evenly distributed. For binary classification, the max impurity is $0.5$ (when $`p=0.5, 0.5`$).



### **Intuition**

* Gini impurity measures how often a **random sample would be misclassified** if assigned randomly based on class proportions.
* A **lower Gini** means higher purity (better split).
* Decision trees use Gini Impurity (or Entropy) to decide **which feature to split on**, aiming to minimize impurity at each step.



### **Practical Use Cases**

* **CART (Classification and Regression Trees)** uses Gini Impurity as the default splitting criterion.
* Popular in **Random Forests** for feature selection and tree construction.
* Useful in **binary and multi-class classification** when building interpretable tree-based models.
* Often preferred over Entropy in large datasets because it’s faster (no logarithms).


**Key Insight**:

* **Gini Impurity vs. Entropy**: Both measure node impurity, but Gini is simpler and tends to produce similar splits. Entropy is more sensitive to class imbalance.

---

