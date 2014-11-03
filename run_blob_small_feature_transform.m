
%% Find lambda for penalized ridge regression

clear all
close all;

[X_train, y_train, X_test, xte] = temp_load_regression_data(1);
N = size(X_train,1);

alpha = 0.1;
K = 10;

 lambda = 1e-4;
 d=2;
 Xp = myPoly(X_train(:,1:36), d);
 tX = [ones(N,1) Xp X_train(:,37:end)]; 
[meanTrainError, meanValidationError, beta]= KfoldCV_updated(K, tX, y_train, 'ridgeRegression', alpha, lambda);
 
save('beta_degree2.mat', 'beta')

mean(meanTrainError)
std(meanTrainError)
mean(meanValidationError)
std(meanValidationError)

 fprintf(1, 'Train and Validation error %3.3f, %3.3f\n', mean(meanTrainError), mean(meanValidationError));