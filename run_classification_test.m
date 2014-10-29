
%% loads the data
%% transforms categorical features in dummyvars
%% normalizes X_train to 0 mean and 1 std
%% applies the same transformation to X_test

%%TODO ? remove outliers fot this data also?

[X_train, y_train, X_test] = load_curated_classification_data;

N = size(X_train,1);
tX = [ones(N,1) X_train];

alpha = 0.1;
beta = logisticRegression(y_train, tX, alpha);
predicted_probability = zeros(N, 1);
predicted_probability(sigmoid(tX*beta) >= 0.5) = 1;
train_accuracy_1 = mean(double(predicted_probability == y_train)) * 100;
fprintf('Train Accuracy: %f\n', train_accuracy_1);

lambda = 0.001;
beta = penLogisticRegression(y_train, tX, alpha, lambda);
predicted_probability = zeros(N, 1);
predicted_probability(sigmoid(tX*beta) >= 0.5) = 1;
train_accuracy_2 = mean(double(predicted_probability == y_train)) * 100;

fprintf('Train Accuracy normal logistic: %f\n', train_accuracy_1);
fprintf('Train Accuracy penalized logistic: %f\n', train_accuracy_2);