%% Find parameters alpha, initial beta for gradient search

%% Find lambda for penalized logistic regression

clear all
[X_train, y_train, X_test] = load_regression_data;
close all;
N = size(X_train,1);
tX = [ones(N,1) X_train];

alpha = 0.1;
lambda = [0,1e-7, 1e-6,1e-5, 0.0001,0.001,0.01,0.1,1];
meanTrainError = zeros(length(lambda),1);
meanValidationError = zeros(length(lambda),1);
K = 5;
for i=1:length(lambda)
 [meanTrainError(i), meanValidationError(i)]= KfoldCV(K, tX, y_train, 'ridgeRegression', alpha, lambda(i));
 fprintf(1, 'Train and Validation error %3.3f, %3.3f\n', meanTrainError(i), meanValidationError(i));
end

plot(sqrt(meanTrainError), '*');
hold on;
plot(sqrt(meanValidationError),'o');

fprintf(1, 'Mean Train and Validation error %3.3f, %3.3f\n', mean(meanTrainError), mean(meanValidationError));


DP = 10;
meanTrainError = zeros(DP,1);
meanValidationError = zeros(DP,1);
 
K = 5;

%% best d is 5
 for d =1:DP
       lambda =1e-7;
       Xp = myPoly(X_train(:,1:43), d);
       tX = [ones(N,1) Xp X_train(:,44:end)];
       %tX = [ones(N,1) myPoly(X_train,d)];
      [meanTrainError(d), meanValidationError(d)]= KfoldCV(K, tX, y_train, 'ridgeRegression', alpha, lambda);
 
 end  
 close all
  figure
 plot(sqrt(meanTrainError), '*');
hold on;
plot(sqrt(meanValidationError),'o');
    meanTrainError
    meanValidationError
 fprintf(1, 'Train and Validation error %3.3f, %3.3f\n', mean(meanTrainError), mean(meanValidationError));
