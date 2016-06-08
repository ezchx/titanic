%% neural network
% adapted from Andrew Ng's machine learning course



%% =========== Part 0: Initialization =============
clear ; close all; clc

Theta1 = [];
Theta2 = [];

%% Setup the parameters you will use for this exercise
%input_layer_size  = 400;  % 20x20 Input Images of Digits
%hidden_layer_size = 25;   % 25 hidden units
%num_labels = 10;          % 10 labels, from 1 to 10

input_layer_size  = 16;
hidden_layer_size = 36;
num_labels = 1;




%% =========== Part 1: Load Data =============
% Load Training Data
%load('ex4data1.mat');
%m = size(X, 1);

train_data = load('titanic_train_data.txt');

% split data into training and cross validation data sets

cv_data = [];
cv_data_length = round(size(train_data, 1) * 0.2);

for i = 1:cv_data_length
  cv_sample_row = randi(size(train_data, 1));
  cv_data = [cv_data; train_data(cv_sample_row, :)];
  train_data(cv_sample_row, :) = [];
end

X = train_data(:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
y = train_data(:, 17);

Xcv = cv_data(:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
ycv = cv_data(:, 17);

test_data = load('titanic_test_data.txt');
Xtest = test_data(:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);




%% ================ Part 2: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];





%% =================== Part 3: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 750);

%  You should also try different values of lambda
lambda = 0.3;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));





%% ================= Part 4: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

%disp("Theta1");
%fprintf('%f\n', Theta1);
%disp(" ");
%disp("Theta2");
%fprintf('%f\n', Theta2);

pred_train = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_train == y)) * 100);

pred_cv = predict(Theta1, Theta2, Xcv);
fprintf('\nCross Validation Set Accuracy: %f\n', mean(double(pred_cv == ycv)) * 100);

pred_test = predict(Theta1, Theta2, Xtest);

save pred_test.txt pred_test;

