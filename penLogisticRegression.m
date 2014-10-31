function beta = penLogisticRegression ( y, tX, alpha, lambda )
% Summary of this function goes here
%   Detailed explanation goes here

  % Initialize algorithm parametes
  maxIters = 1000;
  beta = randn( size(tX, 2), 1 );

  for k = 1 : maxIters
      [cost, grad] = computeCostGradLogisticRegression(y, tX, beta, lambda);
      beta = beta - alpha * grad;
      fprintf(1, 'current cost %3.3f\n', cost);
  end
  
end