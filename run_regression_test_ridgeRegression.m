%% Pre-processing Part
% Loading the data
close all
clear all
clc
data = load('Rome_regression.mat');
D = size(data.X_train,2);
lambda = 1e-5;

% Plotting the data before processing
figure;
hist(data.y_train);
line([4500,4500],[0,500], 'Color','r', 'LineWidth', 3)
title('Distribution of y training values')


%% Processing - Training Part
% Remove outliers
indices = (data.y_train < 4500);
y_train_all = data.y_train(indices);
X_train_all = data.X_train(indices,:);

% Keep randomly 75% of the data for training
percentage = 0.75;
N = round(percentage * size(X_train_all,1));
ind_train = randperm(size(X_train_all,1));
ind_train = ind_train(1:N);
% Remove the last features from the array
X_train = X_train_all(ind_train,1:(D-7));
y_train = y_train_all(ind_train,1);

figure;
hist(y_train);

% Compute mean and std of the training set
X_mean = mean(X_train);
X_std = std(X_train);

% Normalize the data to have 0 mean and 1 std
X_mean_rep = repmat(X_mean,[N, 1]);
X_std_rep = repmat(X_std,[N,1]);
X_train_normalised = X_train - X_mean_rep;
X_train_normalised = X_train_normalised ./ X_std_rep;

% Try mean fitting
beta0 = mean(y_train);
err = computeCostMSE(y_train, ones(size(y_train))*beta0, 1);
err = sqrt(err);
fprintf(1,'RMSE using LeastSquares mean regression %3.3f \n', err);

% Try ridge regression
tX = [ones(size(y_train)) X_train_normalised];
betaLS = ridgeRegression(y_train, tX, lambda);
rmseTr = computeCostMSE(y_train, tX, betaLS);
rmseTr = sqrt(rmseTr);
fprintf(1,'RMSE using LeastSquares ridge regression %3.3f \n', rmseTr);


%% Processing - Validation Part
% Try the model on the validation data
ind_test = find(ismember(1:size(X_train_all,1), ind_train)==0);
X_valid = X_train_all(ind_test,1:(D-7));
y_valid = y_train_all(ind_test,1);

% Normalize it with the same mean and std as the training data
N = size(X_valid,1);
X_mean_rep = repmat(X_mean,[N, 1]);
X_std_rep = repmat(X_std,[N,1]);
X_valid_normalised = X_valid - X_mean_rep;
X_valid_normalised = X_valid_normalised ./ X_std_rep;

% Error on validation set using mean prediciton
err = computeCostMSE(y_valid, ones(size(y_valid))*beta0, 1);
err = sqrt(err);
fprintf(1,'RMSE Validation using LeastSquares mean prediction %3.3f \n', err);

% Error on the validation using ridge prediction
tX = [ones(size(y_valid)) X_valid_normalised];
rmseTrVa = computeCostMSE(y_valid, tX, betaLS);
rmseTrVa = sqrt(rmseTrVa);
fprintf(1,'RMSE Validation using LeastSquares ridge prediction %3.3f \n', rmseTrVa);
