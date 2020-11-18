# SVM

### About the Dataset
In this project I am using a synthetic dataset . It has 450 data points on a 2-D plane. The first two columns contain the X and Y attribute values and the third column contains the class label. 

 What is SVM ?
**Support Vector Machine** is a Machine Learning algorithm used to find a hyperplane in the N-dimensional hyperspace to classify the data points.

### Learning a Linear SVM

The linear SVM learns an oblique linear boundary along with a margin surrounding it. The points
on the margin are called as the support vectors. The SVM maximizes its boundaries on both sides to avoid overfitting. But there must be a limit to define the extent of the boundary. So, we use the Soft Margin Approach by assigning a penalty parameter C to the error term. In this dataset after increasing the value of the penalty parameter to C= 2000 we get the best results for accuracy , precision and recall.

### Plots for Linear SVM

![Linear SVM](https://raw.githubusercontent.com/akshataupadhye/SVM/main/linear_svm.png)

### Learning a Non Linear SVM
SVM learns a boundary by defining a margin around it. The margins are defined using the subset of data points known as support vectors. The SVM is a maximal margin classifier which extends its boundary to overcome the problem of overfitting. But to give a limit to the margin extension the soft margin approach is used in which a penalty parameter C is introduced. In this dataset for the nonlinear SVM after tuning the penalty parameter C =2000 the best results were observed.

### Plots for Non Linear SVM

![Linear SVM](https://raw.githubusercontent.com/akshataupadhye/SVM/main/nl_svm.png)

## Results and Analysis

### Comparison of the Decision Boundaries of Decision Tree ,Ideal Decision tree, Linear SVM and Non Linear SVM:

The Linear SVM has a low accuracy value of 0.88 when compared with decision tree and the nonlinear SVM. The SVM follows a maximal margin approach. Even after tuning the penalty parameter for error which is C the metrics given by the linear SVM is not very accurate. Even the precision and recall values are comparatively lower. Looking at the boundaries on the plot it can be clearly seen that even after maximizing the margin of the SVM there are data points being misclassified.

The Non-Linear SVM with a ‘rbf’ Kernel has the best performance for the data set. Looking at the
performance metrics we can see that accuracy, precision and recall values are all 1. It is giving the exact classification for all data points. Looking at the plots we can see that the boundaries drawn to separate the data sets is very clear
