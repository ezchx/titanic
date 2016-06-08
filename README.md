Kaggle Titanic project

All files were adapted from Andrew Ng's machine learning course.

All models used the same 16 variables.

![](https://github.com/ezchx/titanic/blob/master/fit_table2.png)

Logistic Regression - my best training result was an 84.1% fit with 500 runs and lambda = 0. Surprisingly, this gave worst result with the Kaggle test data (78%).

Linear Regression - gradient descent equaled the normal equation after 500K runs with an alpha of 0.1. The Kaggle test results were in the top 20%.

![](https://github.com/ezchx/titanic/blob/master/train_vs_cv_chart.png)

Neural Network - most of my models trained around 86%. The best CV data was based on 36 hidden neurons and 750 training runs. 81.3 test results were in the top 9%.

All hail Professor Ng!
