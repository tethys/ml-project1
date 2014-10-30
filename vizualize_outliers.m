
close all;
clear all;
data = load('Rome_regression.mat');

y_train = data.y_train(:,1);

D = size(data.X_train,2) - 7;
X_train_real = data.X_train(:,1:D);


%% Make categorical data into dummy variables
X_categorical = data.X_train(:, D+1:end) + 1;
X_dummy_var = dummyvar(X_categorical);
X_train = [X_train_real X_dummy_var];


%% Compute mean and std of the training set
X_mean = mean(X_train);
X_std = std(X_train);
%% Normalize the data to have 0 mean and 1 std
N = size(X_train,1);
X_mean_rep = repmat(X_mean,[N, 1]);
X_std_rep = repmat(X_std,[N,1]);
X_train_normalised = X_train - X_mean_rep;
X_train_normalised = X_train_normalised ./ X_std_rep;


figure;
scatterhist(X_train_normalised(:,36), y_train);
hold on;
plot([-3,5],[4800,4800],'LineWidth',2, 'Color','r')
plot([1.4,1.4],[0,7000],'LineWidth',2, 'Color','k')

sum(X_train_normalised(:,36) > 1.4)