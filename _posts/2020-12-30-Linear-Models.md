---
date: 2020-12-30
title: Linear Models in Machine Learning
image: /assets/img/post3/lineer_denklem.jpg
categories: [Machine Learning]
tags: [Algorithms, Supervised Learning]
---

Linear models are among the most commonly used and basic techniques in machine learning. They are actually based on a concept familiar to everyone: <b>linear functions</b>. For instance, if we write $$\hat{y} = W_i X + b_i $$, it may seem complex at first. However, if we write it as $$y = ax + b$$, we can see that it's essentially the same structure. In this case, W and b are the parameters we control, while X is the independent variable and y is the dependent variable of the equation.

> X: Inputs <br>
> ŷ: Outputs (Predicted values) <br>
> W: Weight values <br>
> b: Bias value <br>


In this equation, our goal is to compute the value of y as accurately as possible to make precise predictions. Since we do not control X (it is provided externally), we can only adjust the values of W and b to achieve the most accurate result for y. Finding the optimal values for W and b is the primary task of linear models. Linear models can be used in both regression and classification tasks. For instance, a well-known example of regression is predicting house prices when buying or selling. After quantifying the house's features such as square footage, number of rooms, and neighborhood, a model is trained to find the optimal W and b values and provide a price estimate based on the given features. There are several types of regression models including Linear Regression, Ridge Regression, and Lasso Regression.

# Regression

![LineerRegression](/assets/img/post3/lineer_regresyon.png)
_Lineer Regression_

## Linear Regression

Linear regression is the simplest form of regression that uses linear equations. The formula for a linear regression model is:

> $$\hat{y} = w_0 x_0 + w_1 x_1 + \ldots + w_n x_n + b$$

Simplifying this formula as described above:

> $$\hat{y} = w_0 x_0 + b$$

Here, $$w_0$$ represents our slope, and b is the y-intercept. The difference between the predicted and actual points is computed using the squared errors. The sum of these squared differences across all points is calculated. This value is known as the Mean Squared Error (MSE) in literature, which is one of the most standard methods. The objective is to minimize the MSE as much as possible to achieve the best fit.

## Ridge Regression

Ridge regression uses the same calculation method as linear regression, employing MSE. Unlike linear regression, Ridge regression aims to shrink the weights (W values) towards zero. To ensure predictions can still be made, a bias (b) value is used. This process is known as regularization, and the method used in Ridge regression is called L2 Regularization. L2 regularization works similarly to the squared error, MSE.

## Lasso Regression

Lasso regression operates under a similar principle to Ridge regression. However, it uses L1 regularization metric instead of L2. L1 regularization is computed as the sum of the absolute values of the differences between points, rather than the squared differences.

<b>Linear models can also be used for <i>classification</i> tasks.</b>

## Classification

For binary classification, for example, we can set up a linear model as follows:

> $$\hat{y} = w_0 x_0 + w_1 x_1 + \ldots + w_n x_n + b > 0$$

As seen, this formula closely resembles linear regression. However, it includes a threshold value. If the computed value is greater than 0, the model returns +1; otherwise, it returns -1. Linear models can be classified into different types based on the regularization algorithms and measurement metrics they use. The same applies to classification tasks. Two well-known methods are Logistic Regression and Support Vector Machines (SVMs).
Logistic Regression

Despite the name, logistic regression should not be confused with regression methods. Logistic regression is a classification algorithm.

> <b>Note:</b><i> In problems with multiple features, one feature may be selected and the others separated for solving the problem.</i>


# References:
- Andreas C. Müller,Sarah Guido, Introduction to Machine Learning with Python.(2017)
