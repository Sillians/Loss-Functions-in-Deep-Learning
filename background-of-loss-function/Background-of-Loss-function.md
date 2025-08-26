# Background of Loss Function

The loss function is a fundamental component in training `machine` and `deep learning models`, as it quantifies the difference between `predictions` and `target` results while providing a clear metric for performance evaluation. During `backpropagation`, the gradient of the loss function guides parameter updates, thereby minimizing errors and improving model accuracy. An effective loss function also plays a role in balancing `bias` and `variance`, helping to prevent overfitting.

Loss functions are typically categorized by the tasks they address. For example, in tabular data, they are commonly used for `regression` and `classification`, while in computer vision, they are essential for tasks such as `image classification`, `object detection`, `recognition`, or `generation for image processing`. For regression tasks, models predict continuous output values, while for classification tasks, they produce discrete labels corresponding to dataset classes. The choice of loss function depends on the type of problem, and standard loss functions are typically aligned with either regression or classification tasks, to ensure effective model training and accurate predictions.


For computer vision tasks, the distinction between loss functions for `discriminative` (e.g., image classification) and `generative` (e.g., image synthesis from different inputs) tasks lies in their objectives and evaluation strategies. In discriminative tasks such as `image classification`, loss functions penalize incorrect predictions to improve accuracy in categorizing images into predefined classes. In contrast, `generative tasks` focus on producing new, realistic outputs that resemble training data. For example, `Generative Adversarial Networks (GANs)` use adversarial loss, where the generator is trained to create images that can fool a discriminator, while `autoencoders` rely on reconstruction loss to ensure generated outputs closely resemble the original inputs. Thus, classification loss functions emphasize prediction accuracy, whereas generative loss functions prioritize `realism` and `fidelity to real data`.

![Alt text](/images/figure-2.png)

---

## Loss Function: Impact, Utilities, and History

The evolution of loss functions reflects the transformative progress of statistical methods and machine learning over the past century. Serving as critical components of model training, loss functions provide quantitative measures of how closely a `model’s predictions` align with `actual outcomes`. A historical perspective reveals the transition from simple statistical techniques to sophisticated deep learning paradigms, highlighting the increasing complexity and specialization of loss functions designed to address diverse challenges across domains. By tracing this development, we gain deeper insight into the foundational principles of modern machine learning and the methodologies that enable effective predictive modeling.


- **StatisticalMethods(19th-20thCentury):** 

The concept of loss functions in statistical methods can be traced back to `Ordinary Least Squares (OLS)`, used in linear regression. OLS aims to minimize the sum of squared difference between observed values and predicted values. This foundational approach serves as an early example of using mathematical functions to minimize error and make predictions based on data.



- **EarlyMachineLearning(Mid-20thCentury):**

As machine learning began to develop, `Maximum Likelihood Estimation (MLE)` became prominent. MLE is a statistical method used to find parameter values that maximize the likelihood of the observed data fitting a particular model. This often involves using loss functions like `Negative Log-Likelihood` to optimize statistical models. 

We can find also `Mean Squared Error (MSE)` which is a standard criterion in statistical modeling and machine learning to evaluate the performance of forecasting and predictive models. `MSE` is sensitive to outliers because it squares the errors, meaning larger errors have a disproportionately high effect on the `MSE` value.


- **Perceptrons and Linear Models (1950s-1980s):** 

The perceptron, an early type of neural network, used a simple loss aimed at minimizing classification errors. In parallel, logistic regression introduced `log loss (cross-entropy loss)`, which measures the `discrepancy` between `actual labels` and `predicted probabilities`. This period marks a shift toward optimizing functions to improve classification accuracy.



- **Artificial Neural Networks (1980s-1990s):**

With the development of neural networks, `Mean Squared Error(MSE)` became commonly used for regression tasks. MSE calculates the `average squared differences between predicted and actual values`, guiding adjustments in model training to minimize errors and enhance prediction precision.



- **SupportVectorMachinesandDeepLearning(1990s-Present):**

The introduction of `Support Vector Machines (SVMs)` brought `hinge loss`, which maximizes the margin between `classes` for classification tasks. In deep learning, `cross-entropy loss` grew in popularity, effectively handling `multi-class classification` by measuring dissimilarity between `predicted probabilities` and `actual class`.


- **Modern Advances (2010s-Present):**

In recent years, new paradigms like `Generative Adversarial Networks (GANs)` have emerged, using adversarial losses to train models in competitive settings. 

`Variational Autoencoders (VAEs)` combine reconstruction loss with `Kullback-Leibler (KL) divergence` for learning latent representations.  `Reinforcement learning` utilizes reward signals as a form of loss to optimize decision-making policies.


![Alt text](/images/figure-3.png)


Loss functions are fundamental to training deep learning models, as they play a central role in making the learning process effective. Their impact extends to both `parameter updates` and `performance evaluation`. During training, the loss function provides direction for updating model parameters `(weights and biases)`. The gradient of the loss with respect to these parameters indicates the magnitude and direction of change needed to reduce errors. Additionally, loss functions serve as metrics to assess model performance, enabling comparisons between models and guiding adjustments to the training process. The choice of loss function is therefore critical: a well-selected function can lead to faster convergence and improved prediction accuracy.

---

## Previous reviews and surveys

This section reviews existing surveys and studies on loss functions, highlighting their scope, organization, and limitations. Prior surveys can generally be grouped by task or technique. Many have concentrated on computer vision, particularly image and `semantic segmentation`, `object detection`, `face recognition`, and related applications. These works emphasize the critical importance of selecting suitable loss functions to boost model performance and accuracy in task-specific contexts. Some surveys also introduce new loss functions to address persistent challenges such as  `data imbalance`, `anatomical precision`, and `domain-specific requirements`.


Beyond application-focused studies, several surveys categorize and analyze loss functions based on methodological perspectives, such as traditional machine learning versus deep learning. These reviews typically organize loss functions into taxonomies, examine their theoretical underpinnings, and assess their computational properties. Collectively, they aim to guide researchers and practitioners in identifying the most effective loss functions for their specific tasks, while also drawing attention to unresolved issues such as `robustness`, `evaluation standards`, and `scalability`.


A recurring theme across prior works is the dual role of loss functions: as tools for optimization and as evaluation metrics. Some surveys have taken a broad approach, systematically cataloging dozens of loss functions across `regression`, `classification`, `unsupervised learning`, and `generative modeling`. Others have been more specialized, focusing on the nuances of deep learning applications, where loss functions are often paired with performance measures to assess both training dynamics and predictive outcomes.


In contrast to these prior surveys, the proposed review provides a broader and more integrative perspective. It examines commonly used loss functions not only in computer vision but also in `tabular data prediction` and `time series forecasting`. The review highlights advanced `deep learning architectures` such as `CNNs`, `Vision Transformers`, and `diffusion models`, analyzing how different loss functions are employed individually or in combination to optimize model training. The scope of tasks covered ranges from 
- `regression` and 
- `classification` in structured data to 
- `image classification`, 
- `object detection`, 
- `semantic segmentation`, 
- `action recognition`, 
- `object tracking`, and 
- `generative modeling` in vision.












