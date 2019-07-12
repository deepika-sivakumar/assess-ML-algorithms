# Assess Machine Learning Algorithms

### Problem Description:
To implement Decision Tree Learner, Random Tree Learner and Bootstrap Aggregating Learner from scratch for Prediction problems(using feature selection methods).

### Dataset: 
Various stock index returns from Istanbul Stock Exchange from the UCI Machine Learning Data Repository.
<br/> https://archive.ics.uci.edu/ml/datasets/ISTANBUL+STOCK+EXCHANGE

| Variable | Column Name in Dataset | Denotes |
| -------- |:----------------------:| ----------------- |
| X - Feature | ISE-TL | Istanbul stock exchange national 100 index (Currency: Turkish lira) |
| X - Feature | ISE-USD | Istanbul stock exchange national 100 index (Currency: USD) |
| X - Feature | SP | S&P 500 index |
| X - Feature | DAX | Stock market return index of Germany |
| X - Feature | FTSE | Stock market return index of UK |
| X - Feature | NIKKEI | Stock market return index of Japan |
| X - Feature | BOVESPA | Stock market return index of Brazil |
| X - Feature | EU | MSCI European index |
| Y - Predict | EM | MSCI emerging markets index |

Our goal is to predict the MSCI emerging markets index based on other index returns.

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
