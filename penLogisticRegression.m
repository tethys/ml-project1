function  beta = penLogisticRegression ( y, tX, alpha, lambda )
%
% 'beta' computation for the method logistic regression
%
% Mandatory inputs:
% y      : y of the training dataset
% tX     : X of the training dataset
% alpha  : method parameter alpha
% lambda : method parameter lambda
%
% Outputs:
% beta   : estimated coefficients of the model
%

  % Initialize algorithm parametes
  maxIters = 1000;
  beta = zeros( size(tX, 2), 1 );
  err = 1 ./ eps;
  [Lold, ~] = computeCostGradLogisticRegression( y, tX, beta, lambda );
  k = 1;
  
  % Gradient descent iteration
  while ( k <= maxIters && err > eps )
        
    % Gradient computation
    [~, grad] = computeCostGradLogisticRegression( y, tX, beta, lambda );
    
    % Updating for value of 'beta'
    beta = beta - alpha .* grad;
    
    % Cost computation
    [cost, ~] = computeCostGradLogisticRegression( y, tX, beta, lambda );    
    err = abs(Lold - cost);
    Lold = cost;
    
    k = k + 1;
    
  end
  
end