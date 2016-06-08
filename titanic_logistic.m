%% Logistic Regression with Regularization

%% Initialization
clear ; close all; clc

%% Load Data
%  The first 10 columns contains the X values and the 11th column
%  contains the label (y).

data = load('titanic_train_data.txt');
X = data(:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]); y = data(:, 17);

% Add intercept term to x
m = length(X(:,1));
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 0.0;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 500);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% print theta
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);



