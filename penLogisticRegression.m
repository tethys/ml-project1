function beta = penLogisticRegression(y,tX,alpha,lambda)
% Summary of this function goes here
%   Detailed explanation goes here

  % Initialize algorithm parametes
  maxIters = 1000;
  beta = randn( size(tX, 2), 1 );
  
  err = 1/eps;
  
  [Lold, ~] = computeCostGradLogisticRegression(y, tX, beta, lambda);
    % Gradient descent iteration
    k = 1;
  while ( k <= maxIters && err > eps )
        
    % Gradient computation
    [~, grad] = computeCostGradLogisticRegression(y, tX, beta, lambda);
    % Updating for value of 'beta'
    beta = beta - alpha .* grad;
    
    [cost, ~] = computeCostGradLogisticRegression(y, tX, beta, lambda);
    
    % Error computation
    err = abs(Lold - cost);
    Lold = cost;
    
    k = k + 1;
    
  end
end