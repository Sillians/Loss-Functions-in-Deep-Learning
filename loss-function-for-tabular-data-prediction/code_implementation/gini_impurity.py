import numpy as np

def gini_impurity(y):
    """
    Calculate the Gini Impurity for a list of class labels.

    Parameters:
    y (list or numpy array): List of class labels.

    Returns:
    float: Gini Impurity value.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    impurity = 1.0 - sum((count / total) ** 2 for count in counts)
    return impurity

# Example usage:
if __name__ == "__main__":
    labels = [0, 1, 0, 1, 1, 0, 0]
    print("Gini Impurity:", gini_impurity(labels))