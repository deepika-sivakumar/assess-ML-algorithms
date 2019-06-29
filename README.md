# Assess Machine Learning Algorithms

### Problem Description:
To implement Decision Tree Learner, Random Tree Learner and Bootstrap Aggregating Learner for Prediction problems(using feature selection methods).

### Dataset: 
To predict the returns for MSCI Emerging Markets (EM) index based on previous index returns from the UCI Machine Learning Data Repository.

### Solving the problem:
* Import data and split data for training and testing
* Framing the prediction problem as a **Regression problem** to be solved.
* Implement each learner as a separate class.
* Train and test each model.
* Run experiments and make data visualization of results.

### Experiments & Results:
* The leaf size of a learner had an impact in overfitting. 
* **Bootstrapping**/Bagging significantly reduced **overfitting**, but did not eliminate it altogether. 
* Decision trees outperformed Random trees in terms of **accuracy**(Metric: Mean Absolute Error-MAE), whereas Random trees had better **computational time**.

### Best Practices:
* **Programming from scratch:** Implemented supervised learning algorithms from scratch to understand the underlying workings.
* **Object Oriented Programming:** Employed object-oriented approach, where each learner is a class, for which we can create objects and is also provides inheritence(reuse code between learners) and abstraction(The learner can be called on a higher level giving input data and getting back predictions)
