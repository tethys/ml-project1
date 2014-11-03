function predict_regression_values(X_test, y_test, xte)

%% X_test should be processed
 m1_indices = xte(:,36) <= -14;
 m2_indices = xte(:,36) > -12;


 
 beta_0 = load('beta_degree3.mat');
 beta_1 = load('beta_degree2.mat');
 
 
 tX1 = process_input_model1(X_test(m1_indices,:));
 tX2 = process_input_model2(X_test(m2_indices,:));
 
 size(tX2)
 size(beta_1.beta)
 computeCostRMSE(y_test(m1_indices), tX1,beta_0.beta)
 computeCostRMSE(y_test(m2_indices), tX2,beta_1.beta)
  err1 = y_test(m1_indices) - tX1*beta_0.beta;
  err2 = y_test(m2_indices) - tX2* beta_1.beta;
  
 totrmse =  err1'*err1 + err2'*err2;
 totrmse =  sqrt(totrmse/length(y_test))

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