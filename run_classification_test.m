
%% loads the data
%% transforms categorical features in dummyvars
%% normalizes X_train to 0 mean and 1 std
%% applies the same transformation to X_test

%%TODO ? remove outliers fot his data also?

[X_train, y_train, X_test] = load_curated_classification_data;

N = size(X_train,1);
tX = [ones(N,1) X_train];

beta = logisticRegression(y_train, tX);

m = size(tX, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(N, 1);
p(sigmoid(tX*beta) >= 0.5) = 1;

fprintf('Train Accuracy: %f\n', mean(double(p == y_train)) * 100);