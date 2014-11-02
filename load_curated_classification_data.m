function [X_train, y_train, X_test] = load_curated_classification_data


clear all
data = load('Rome_classification.mat');

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

categorical_data = X_test(:,14);
temp = dummyvar(categorical_data);
X_test = [X_test(:,1:13) X_test(:,15:end) temp];

%% Compute mean and std of the training set
X_mean = mean(X_train);
X_std = std(X_train);


%% Normalize the data to have 0 mean and 1 std
N = size(X_train, 1);
X_mean_rep = repmat(X_mean,[N, 1]);
X_std_rep = repmat(X_std,[N,1]);
X_train = X_train - X_mean_rep;
X_train = X_train./ X_std_rep;

%% Do the same for the test data
N = size(X_test, 1);
X_mean_rep = repmat(X_mean, [N,1]);
X_std_rep = repmat(X_std, [N,1]);
X_test = X_test - X_mean_rep;
X_test = X_test ./ X_std_rep;

indices = y_train == -1;
y_train(indices) = 0;

size(y_train)

%% Remove outliers from the train data
% If the data are normally distributed with mean 0 and std 1, then the
% 99.99% of them are appear between the values -3.891 and 3.891
[row_to_remove, ~] = find(X_train < -7.5);
[temp, ~] = find(X_train > 7.5);
row_to_remove = unique([row_to_remove ; temp]);
row_to_remove = sort(row_to_remove,'descend');
X_train(row_to_remove,:)=[];
y_train(row_to_remove,:)=[];

%boxplot(X_train);
size(y_train)


end