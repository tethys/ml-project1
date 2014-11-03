%% Find parameters alpha for gradient search

close all
clear all
clc
[X_train, y_train, ~] = load_regression_data(1);
close all
N = size(X_train,1);
tX = [ones(N,1) X_train];

alpha = [1e-7,1e-5,0.001,0.01,0.1,1];
meanTrainError = zeros(length(alpha),1);
meanValidationError = zeros(length(alpha),1);
K = 5;
for i=1:length(alpha)
 [meanTrainError(i), meanValidationError(i)]= KfoldCV(K, tX, y_train, 'leastSquares', alpha(i));
 fprintf(1, 'Train and Validation error %3.3f, %3.3f\n', meanTrainError(i), meanValidationError(i));
end

plot(sqrt(meanTrainError), '*');
hold on;
plot(sqrt(meanValidationError),'o');

fprintf(1, '\nMean Train and Validation error %3.3f, %3.3f\n\n', mean(meanTrainError), mean(meanValidationError));
[min_tr, min_tr_ind] = min(meanTrainError);
[min_val, min_val_ind] = min(meanValidationError);
fprintf(1, 'Minimum Train error %3.3f for alpha = %f\n', min_tr, alpha(min_tr_ind) );
fprintf(1, 'Minimum Validation error %3.3f for alpha = %f\n\n', min_val, alpha(min_val_ind) );


%% Find lambda for penalized ridge regression

clear all
close all;

[X_train, y_train, X_test] = load_regression_data(1);

N = size(X_train,1);
tX = [ones(N,1) X_train];

alpha = 0.1;
lambda = [1e-7,1e-5, 0.0001,0.001,0.01,0.1,1];
meanTrainError = zeros(length(lambda),1);
meanValidationError = zeros(length(lambda),1);
K = 2;
for i=1:length(lambda)
 [meanTrainError(i), meanValidationError(i)]= KfoldCV(K, tX, y_train, 'ridgeRegression', alpha, lambda(i));
 fprintf(1, 'Train and Validation error %3.3f, %3.3f\n', meanTrainError(i), meanValidationError(i));
end

plot(meanTrainError, '*');
hold on;
plot(meanValidationError,'o');

fprintf(1, '\nMean Train and Validation error %3.3f, %3.3f\n\n', mean(meanTrainError), mean(meanValidationError));
[min_tr, min_tr_ind] = min(meanTrainError);
[min_val, min_val_ind] = min(meanValidationError);
fprintf(1, 'Minimum Train error %3.3f for lamdda = %f\n', min_tr, lambda(min_tr_ind) );
fprintf(1, 'Minimum Validation error %3.3f for lambda = %f\n\n', min_val, lambda(min_val_ind) );


DP = 4;

 
K = 15;
meanTrainError = zeros(DP,1);
meanValidationError = zeros(DP,1);

% best d is 5
 for d =1:DP
       lambda = 1e-2;
       alpha = 0.1;
     %  Xp = myPoly(X_train(:,1:43), d);
       Xp = myPoly(X_train(:,1:35), d);
       Xe = myExp(X_train(:,1:35), d);
       tX = [ones(N,1) Xp X_train(:,37:end)]; %X_train(:,36:end)];% X_train(:,44:end)];
       %tX = [ones(N,1) myPoly(X_train,d)];
      [meanTrainError(d), meanValidationError(d)]= KfoldCV(K, tX, y_train, 'ridgeRegression', alpha, lambda);
 
 end  
 close all
  figure
 plot(1:1:DP, meanTrainError(1:end), '*-');

hold on;
plot(1:1:DP, meanValidationError(1:end),'ro-');
hx = xlabel('degree of polynomial');
hy = ylabel('mean RMSE');

    meanTrainError
    meanValidationError
     h = gca;
h.XTick = 2:1:10;
    pause
 fprintf(1, 'Train and Validation error %3.3f, %3.3f\n', mean(meanTrainError), mean(meanValidationError));
