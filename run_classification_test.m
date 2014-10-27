
%% loads the data
%% transforms categorical features in dummyvars
%% normalizes X_train to 0 mean and 1 std
%% applies the same transformation to X_test

%%TODO ? remove outliers fot his data also?

[X_train, y_train, X_test] = load_curated_classification_data;

N = size(X_train,1);
tX = [ones(N,1) X_train];

beta = logisticRegression(y_train, tX);