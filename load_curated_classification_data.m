function [X_train, y_train, X_test] = load_curated_classification_data


clear all
data = load('Rome_classification.mat');

hist(data.y_train)
title('Distribution of y training values')
%% We observe that there are twice as many samples from one class type
%% than the other

%%% Remove outliers
y_train = data.y_train;
X_train = data.X_train;
X_test = data.X_test;

%% Feature number 14 is categorical, make it a dummy variable
categorical_data = X_train(:,14);
temp = dummyvar(categorical_data);
X_train = [X_train(:,1:13) X_train(:,15:end) temp];



%% Compute mean and std of the training set
X_mean = mean(X_train);
X_std = std(X_train);


%% Normalize the data to have 0 mean and 1 std
N = size(X_train, 1);
X_mean_rep = repmat(X_mean,[N, 1]);
X_std_rep = repmat(X_std,[N,1]);
X_train = X_train - X_mean_rep;
X_train = X_train./ X_std_rep;

N = size(X_test, 1);
X_mean_rep = repmat(X_mean, [N,1]);
X_std_rep = repmat(X_std, [N,1]);
X_test = X_test - X_mean_rep;
X_test = X_test ./ X_std_rep;

end