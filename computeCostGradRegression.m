%% lambda = 0 is linear regression
%5 lambda > 0 is ridge regression
function [cost, grad] = computeCostGradRegression(y, tX, beta, lambda)

N = length(y);
cost = 1/(2*N)*(tX*beta - y)'*(X*theta - y) + lambda/(2*N)*sum(beta(2:end).^2);

error = y - tX * beta;
grad = -1/N * tX'*error + lambda/N*sum(beta(2:end));

end