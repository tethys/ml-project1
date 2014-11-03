
function yPred = run_Xtest_prediction
clear all;

[~, ~, X_test1, ind_test1] = load_regression_data(0);
[~, ~, X_test2, ind_test2] = load_regression_data(1);

data = load('Rome_regression.mat');


 
beta_0 = load('best_beta_degree3.mat');
beta_1 = load('best_beta_degree2.mat');
 
 
tX1 = process_input_model1(X_test1);
tX2 = process_input_model2(X_test2);
 
size(tX1)
size(tX2)
yPred = zeros(size(data.X_test,1),1);

for i = 1:length(ind_test1)
    yPred(ind_test1(i)) = tX1(i,:)* beta_0.beta;
end
for i = 1:length(ind_test2)
    yPred(ind_test2(i)) = tX2(i,:)* beta_1.beta;
end
%yPred(ind_test1) = tX1 * beta_0.beta;
%yPred(ind_test2) = tX2 *beta_1.beta;

csvwrite('predictions_regression.csv', yPred)

end

function tX1 = process_input_model1(Xtest)
 d=3;
 N = size(Xtest,1);
  size(Xtest)
 Xp = myPoly(Xtest(:,1:36), d);
 tX1 = [ones(N,1) Xp Xtest(:,37:end)];
 size(tX1)
end


function tX2 = process_input_model2(Xtest)
 d=2;
 N = size(Xtest,1);
 Xp = myPoly(Xtest(:,1:36), d);
 tX2 = [ones(N,1) Xp Xtest(:,37:end)]; 
end