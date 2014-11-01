%% Loading data - Fit parameters
% First fitting
close all
clear all
clc

K = 1243;
alpha = 0.1;
[X_train, y_train, X_test_1 ind_test_1] = load_regression_data(0);
tX = [ones(size(y_train)) X_train];

betaLS_1 = leastSquares(y_train, tX);
rmseTr_1 = computeCostRMSE(y_train, tX, betaLS_1);
fprintf(1,'RMSE using LeastSquares multi linear regression for the first fitting: %3.3f\n', rmseTr_1);

[meanTrainError, meanValidationError]= KfoldCV(K, tX, y_train, 'leastSquares', alpha);
fprintf(1, '\nTrain error: %3.3f\nValidation error: %3.3f\n\n', meanTrainError, meanValidationError);

% Second fitting
clearvars X_train y_train

K = 135;
[X_train, y_train, X_test_2 ind_test_2] = load_regression_data(1);
tX = [ones(size(y_train)) X_train];

betaLS_2 = leastSquares(y_train, tX);
rmseTr = computeCostRMSE(y_train, tX, betaLS_2);
fprintf(1,'RMSE using LeastSquares multi linear regression for the second fitting: %3.3f\n', rmseTr);

[meanTrainError, meanValidationError]= KfoldCV(K, tX, y_train, 'leastSquares', alpha);
fprintf(1, '\nTrain error: %3.3f\nValidation error: %3.3f\n\n', meanTrainError, meanValidationError);

%% Prediction of the test data
M = size(X_test_1,1) + size(X_test_2,1);
y_test = zeros(M,1);
tX_test_1 = [ones(size(X_test_1,1),1) X_test_1];
tX_test_2 = [ones(size(X_test_2,1),1) X_test_2];
for i = 1 : length(ind_test_1)
    y_test(ind_test_1(i)) = tX_test_1(i,:) * betaLS_1;
end
for i = 1 : length(ind_test_2)
    y_test(ind_test_2(i)) = tX_test_2(i,:) * betaLS_2;
end

X_test(ind_test_1, :) = X_test_1;
X_test(ind_test_2, :) = X_test_2;
figure;
scatterhist(X_test(:,36), y_test);
