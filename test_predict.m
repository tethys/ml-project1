

a = load('test20.mat')

for i=36:36
   scatterhist(a.X_test(:,i), a.ytest)
   pause
   close
end

predict_regression_values(a.X_test, a.ytest, a.xte)