%% Find parameters alpha, initial beta for gradient search

%% Find lambda for penalized logistic regression

clear all
[X_train, y_train, X_test] = load_curated_classification_data;
close all;
N = size(X_train,1);
tX = [ones(N,1) X_train];

alpha = 0.1;
lambda = [1e-5, 0.0001,0.001,0.01,0.1,10];
meanTrainError = zeros(length(lambda),1);
meanValidationError = zeros(length(lambda),1);
K = 5;
for i=1:length(lambda)
    i
 [meanTrainError(i), meanValidationError(i)]= KfoldCV(K, tX, y_train, 3, alpha, lambda(i));
end

plot(meanTrainError, '*');
hold on;
plot(meanValidationError,'o');