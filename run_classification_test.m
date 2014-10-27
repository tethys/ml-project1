
clear all
data = load('Rome_classification.mat');
D = size(data.X_train,2);

hist(data.y_train)
title('Distribution of y training values')
%% We observe that there are twice as many samples from one class type
%% than the other

%%% Remove outliers
y_train_all = data.y_train;
X_train_all = data.X_train;

categorical_data = X_train_all(:,14);
temp = dummyvar(categorical_data);
X_train_all = [X_train_all(:,1:13) X_train_all(:,15:end) temp];


%% Feature number 14 is categorical, make it a dummy variable

%%% Keep 0.75 of the data for training
percentage = 0.75;
N = round(percentage * size(X_train_all,1));
%%% Remove the last features from the array
X_train = X_train_all(1:N,:);
y_train = y_train_all(1:N,1);


figure

%% Compute mean and std of the training set
X_mean = mean(X_train);
X_std = std(X_train);


%% Normalize the data to have 0 mean and 1 std
X_mean_rep = repmat(X_mean,[N, 1]);
X_std_rep = repmat(X_std,[N,1]);
X_train_normalised = X_train - X_mean_rep;
X_train_normalised = X_train_normalised ./ X_std_rep;


