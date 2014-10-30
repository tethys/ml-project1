function [X_train, y_train, X_test] = load_regression_data

clear all
data = load('Rome_regression.mat');
D = size(data.X_train,2);

%%% Remove outliers
indices = (data.y_train < 4900);
y_train = data.y_train(indices);
X_train = data.X_train(indices,:);
%% Do we remove here outliers??
X_test = data.X_test;
N = size(X_train, 1);


%% Feature number 14 is categorical, make it a dummy variable
categorical_data = X_train(:,D-6:end) + 1;
temp = dummyvar(categorical_data);
X_train = [X_train(:,1:(D-6)) temp];

categorical_data = X_test(:,D-6:end) + 1;
temp = dummyvar(categorical_data);
X_test = [X_test(:,1:(D-6)) temp];

%% Compute mean and std of the training set
X_mean = mean(X_train);
X_std = std(X_train);


%% Normalize the data to have 0 mean and 1 std
X_mean_rep = repmat(X_mean,[N, 1]);
X_std_rep = repmat(X_std,[N,1]);
X_train = X_train - X_mean_rep;
X_train = X_train ./ X_std_rep;

%% Do the same for the test data
N = size(X_test, 1);
X_mean_rep = repmat(X_mean, [N,1]);
X_std_rep = repmat(X_std, [N,1]);
X_test = X_test - X_mean_rep;
X_test = X_test ./ X_std_rep;


