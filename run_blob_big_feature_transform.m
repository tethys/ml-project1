
%% Saves the best beta for the model of blob 1 using
%% polynomial regression of degree 3

[X_train, y_train, X_test, ytest, xte] = temp_load_regression_data(0);
N = size(X_train,1);

save('test20.mat', 'X_test', 'ytest', 'xte')
alpha = 0.1;
trials = 30;
K = 10;

 lambda = 1e-5;
 d=3;
 Xp = myPoly(X_train(:,1:36), d);
 tX = [ones(N,1) Xp X_train(:,37:end)]; 
[meanTrainError, meanValidationError, beta]= KfoldCV_updated(K, tX, y_train, 'ridgeRegression', alpha, lambda);
 
save('beta_degree3.mat', 'beta')

mean(meanTrainError)
std(meanTrainError)
mean(meanValidationError)
std(meanValidationError)
fprintf(1, 'Train and Validation error %3.3f, %3.3f\n', mean(meanTrainError), mean(meanValidationError));