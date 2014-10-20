function beta = leastSquaresGD(y, tX, alpha)
% Summary of this function goes here
%   Detailed explanation goes here


  % algorithm parametes
  maxIters = 1000;
  % initialize
  beta = randn(size(y));

  % iterate
  fprintf('Starting iterations, press Ctrl+c to break\n');


  % INSERT YOUR FUNCTION FOR COMPUTING COST FUNCTION
  L = computeCost(y, tX, beta); 
  k = 1;  
  err = 100;
  while (k < maxIters && err < eps)
    Lold = L;
    % INSERT YOUR FUNCTION FOR COMPUTING GRADIENT 
    g = computeGradient(y, tX, beta);
    L = computeCostMSE(y, tX, beta); 
    
    fprintf('L  %d \n', L);
  
    % INSERT GRADIENT DESCENT UPDATE TO FIND BETA
    beta = beta - alpha.*g;
    err = abs(Lold - L);
    
    k = k + 1;
  end
end

