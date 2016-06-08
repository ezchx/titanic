Kaggle Titanic project

All files were adapted from Andrew Ng's machine learning course and all models used the same 16 variables.

![](https://github.com/ezchx/titanic/blob/master/fit_table2.png)

Logistic Regression - the best training result was 84.1% with 500 runs and lambda = 0. Surprisingly, this gave the worst Kaggle test data result (78%).

Linear Regression - gradient descent converged with the normal equation after 500K runs with an alpha of 0.1. The test result of 79.4% was in the top 20% of all Kaggle submissions.

Neural Network - most of the models trained around 86%. The best CV data was based on 36 hidden neurons and 750 training runs. The test result of 81.3% was in the top 9% of all Kaggle submissions.

![](https://github.com/ezchx/titanic/blob/master/train_vs_cv_chart.png)

All hail Professor Ng!
