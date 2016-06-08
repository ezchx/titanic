%% Titanic gradient descent (linear regression)

%% Clear and Close Figures
clear ; close all; clc

%% Load Data
data = load('titanic_train_data.txt');
X = data(:, 1:16);
y = data(:, 17);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Choose some alpha value
alpha = 0.1;
num_iters = 500000;

% Init Theta and Run Gradient Descent 
theta = zeros(17, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Compute accuracy on our training set
p = (X * theta) >= 0.5;

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);


% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Compute accuracy on normal equation
p = (X * theta) >= 0.5;

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);