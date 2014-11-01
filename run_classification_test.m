
%% loads the data
%% transforms categorical features in dummyvars
%% normalizes X_train to 0 mean and 1 std
%% applies the same transformation to X_test

%%TODO ? remove outliers fot this data also?

[X_train, y_train, X_test] = load_curated_classification_data;

N = size(X_train,1);
tX = [ones(N,1) X_train];
%%

alpha = 0.1;
beta = logisticRegression(y_train, tX, alpha);
% Logistic regression will give you a prediction probability of a test 
% output belonging to class say yn = 1. Call it pHatn. 
pHatn = sigmoid(tX*beta);

% Given this you can assign a class to this test output to 0 or 1 
% (just take max of pHatn and 1-pHatn). Call the class assignment yHatn. 
yHatn = zeros(N, 1);
yHatn(pHatn >= 0.5) = 1;

% Given yhatn and pHatn for all test outputs, you can compute the three errors.
train_RMSE_LR = sqrt(mean((y_train - pHatn).^2));
train_01Loss_LR = mean(double(yHatn ~= y_train));
train_logLoss_LR = mean(-y_train'*log(pHatn) - (1-y_train)'*log(1-pHatn));
%%

lambda = 1.0;
beta = penLogisticRegression(y_train, tX, alpha, lambda);
% Logistic regression will give you a prediction probability of a test 
% output belonging to class say yn = 1. Call it pHatn. 
pHatn = sigmoid(tX*beta);

% Given this you can assign a class to this test output to 0 or 1 
% (just take max of pHatn and 1-pHatn). Call the class assignment yHatn. 
yHatn = zeros(N, 1);
yHatn(pHatn >= 0.5) = 1;

% Given yhatn and pHatn for all test outputs, you can compute the three errors.
train_RMSE_pLR = sqrt(mean((y_train - pHatn).^2));
train_01Loss_pLR = mean(double(yHatn ~= y_train));
train_logLoss_pLR = mean(-y_train'*log(pHatn) - (1-y_train)'*log(1-pHatn));

header ='method,rmse,0-1-loss,log-loss';
fid = fopen('test_errors_classification.csv','wt');
fprintf(fid, '%s\n', header);
fprintf(fid, 'logisticRegression,%f,%f,%f\n', train_RMSE_LR, ...
    train_01Loss_LR, train_logLoss_LR);
fprintf(fid, 'penlogisticRegression,%f,%f,%f\n', train_RMSE_pLR, ...
    train_01Loss_pLR, train_logLoss_pLR);
%%

alpha = 0.2;
beta = logisticRegression(y_train, tX, alpha);
predicted_probability = zeros(N, 1);
predicted_probability(sigmoid(tX*beta) >= 0.5) = 1;
train_accuracy_1 = mean(double(predicted_probability == y_train)) * 100;

lambda = 0.1;
beta = penLogisticRegression(y_train, tX, alpha, lambda);
predicted_probability = zeros(N, 1);
predicted_probability(sigmoid(tX*beta) >= 0.5) = 1;
train_accuracy_2 = mean(double(predicted_probability == y_train)) * 100;

fprintf('Train Accuracy normal logistic: %f\n', train_accuracy_1);
fprintf('Train Accuracy penalized logistic: %f\n', train_accuracy_2);