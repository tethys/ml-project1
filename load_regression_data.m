function [X_train, y_train, X_test, ind_test] = load_regression_data( th )

clear data
data = load('Rome_regression.mat');
D = size(data.X_train,2);

if th == 0
    [indices, ~] = find((data.X_train(:,36) < -13.0) & (data.y_train < 4900));
else
    [indices, ~] = find((data.X_train(:,36) >= -13.0) & (data.y_train >= 4900));
end
y_train = data.y_train(indices);
X_train = data.X_train(indices,:);

%% Make categorical features dummy variables
categorical_data = X_train(:,D-6:end) + 1;
temp = dummyvar(categorical_data);
X_train = [X_train(:,1:(D-7)) temp];

%% Compute mean and std of the training set
X_mean = mean(X_train);
X_std = std(X_train);

%% Remove outliers from the train data
% If the data are normally distributed with mean 0 and std 1, then the
% 99.99% of them are appear between the values mean-3.891*std and mean+3.891*std
low_limit = X_mean - 3.891 * X_std;
high_limit = X_mean + 3.891 * X_std;
row_to_remove = [];
for i = 1 : size(X_train,2)
    [temp, ~] = find(X_train(:,i) < low_limit(i));
    row_to_remove = [row_to_remove ; temp];
end
for i = 1 : size(X_train,2)
    [temp, ~] = find(X_train(:,i) > high_limit(i));
    row_to_remove = [row_to_remove ; temp];
end
row_to_remove = unique(row_to_remove);
row_to_remove = sort(row_to_remove,'descend');
X_train(row_to_remove,:)=[];
y_train(row_to_remove,:)=[];

%% Compute mean and std of the training set
X_mean = mean(X_train);
X_std = std(X_train);

%% Normalize train data to have 0 mean and 1 std
N = size(X_train(:,1:(D-7)), 1);
X_mean_rep = repmat(X_mean(:,1:(D-7)),[N,1]);
X_std_rep = repmat(X_std(:,1:(D-7)),[N,1]);
X_train(:,1:(D-7)) = X_train(:,1:(D-7)) - X_mean_rep;
X_train(:,1:(D-7)) = X_train(:,1:(D-7)) ./ X_std_rep;

%% Do the same for the relevant test data
if th == 0
    [ind_test, ~] = find(data.X_test(:,36) < -13.0);
else
    [ind_test, ~] = find(data.X_test(:,36) >= -13.0);
end
X_test = data.X_test(ind_test,:);

categorical_data = X_test(:,D-6:end) + 1;
if ~isempty(categorical_data)
    temp = dummyvar(categorical_data);
end
X_test = [X_test(:,1:(D-7)) temp];

N = size(X_test(:,1:(D-7)), 1);
X_mean_rep = repmat(X_mean(:,1:(D-7)), [N,1]);
X_std_rep = repmat(X_std(:,1:(D-7)), [N,1]);
X_test(:,1:(D-7)) = X_test(:,1:(D-7)) - X_mean_rep;
X_test(:,1:(D-7)) = X_test(:,1:(D-7)) ./ X_std_rep;

end
