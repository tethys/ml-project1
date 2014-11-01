%% Find parameters alpha, initial beta for gradient search

%% Find lambda for penalized logistic regression

clear all
[X_train, y_train, X_test] = load_curated_classification_data;
close all;
N = size(X_train,1);
tX = [ones(N,1) X_train];

% alpha = 1e-3;
% %alpha = 10.^linspace(-5,3,50);
% %lambda = 10.^linspace(-5,1,100);
% lambda = 10.^linspace(-5,3,50);
% %lambda = [1e-3 1e-5];
% %lambda = [1e-5, 0.0001,0.001,0.01,0.1,10];
% meanTrainError = zeros(length(lambda),1);
% meanValidationError = zeros(length(lambda),1);
% K = 2;
% for i=1:length(lambda)
%  [meanTrainError(i), meanValidationError(i)]= KfoldCV(K, tX, y_train, 'penLogisticRegression', alpha, lambda(i));
%  fprintf(1, 'Lambda %f\nTrain Error %3.5f Validation Error %3.5f\n', lambda(i), meanTrainError(i), meanValidationError(i));
% end
% 
% %lambda(meanTrainError == min(meanTrainError))
% fprintf(1, '\nMean Train and Validation error %3.3f, %3.3f\n\n', mean(meanTrainError), mean(meanValidationError));
% [min_tr, min_tr_ind] = min(meanTrainError);
% [min_val, min_val_ind] = min(meanValidationError);
% fprintf(1, 'Minimum Train error %3.3f for lambda = %f\n', min_tr, lambda(min_tr_ind) );
% fprintf(1, 'Minimum Validation error %3.3f for lambda = %f\n\n', min_val, lambda(min_val_ind) );
% 
% plot(meanTrainError, 'r-','linewidth',2);
% %plot(meanTrainError, '*');
% hold on
% plot(meanValidationError,'b-','linewidth',2);
% grid on;
% legend('Train Error', 'Test Error', 'Location', 'SouthEast');
% %plot(meanValidationError,'o');
% 
% hx = xlabel('Lambda');
% hy = ylabel('Error');
% set([hx, hy], 'fontsize', 11);

%% Find alpha for penalized logistic regression

%alpha = 1e-3;
alpha = 10.^linspace(-5,1,100);
%lambda = 10.^linspace(-5,1,100);
lambda = 1e-5;
%lambda = [1e-3 1e-5];
%lambda = [1e-5, 0.0001,0.001,0.01,0.1,10];
meanTrainError = zeros(length(alpha),1);
meanValidationError = zeros(length(alpha),1);
K = 2;
for i=1:length(alpha)
 [meanTrainError(i), meanValidationError(i)]= KfoldCV(K, tX, y_train, 'penLogisticRegression', alpha(i), lambda);
 fprintf(1, 'Alpha %f\nTrain Error %3.5f Validation Error %3.5f\n', alpha(i), meanTrainError(i), meanValidationError(i));
end

fprintf(1, '\nMean Train and Validation error %3.3f, %3.3f\n\n', mean(meanTrainError), mean(meanValidationError));
[min_tr, min_tr_ind] = min(meanTrainError);
[min_val, min_val_ind] = min(meanValidationError);
fprintf(1, 'Minimum Train error %3.3f for alpha = %f\n', min_tr, alpha(min_tr_ind) );
fprintf(1, 'Minimum Validation error %3.3f for alpha = %f\n\n', min_val, alpha(min_val_ind) );

plot(meanTrainError, 'r-','linewidth',2);
%plot(meanTrainError, '*');
hold on
plot(meanValidationError,'b-','linewidth',2);
grid on;
legend('Train Error', 'Test Error', 'Location', 'NorthEast');
%plot(meanValidationError,'o');

hx = xlabel('Alpha');
hy = ylabel('Error');
set([hx, hy], 'fontsize', 11);