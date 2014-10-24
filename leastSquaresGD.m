function beta = leastSquaresGD( varargin )
% Summary of this function goes here
%   Detailed explanation goes here

  % Chech the arguments
  switch nargin
      case 2
          y = varargin{1};
          tX = varargin{2};
          alpha = 10e-3;
      case 3
          y = varargin{1};
          tX = varargin{2};
          alpha = varargin{3};
      otherwise
          error('Unexpected number of input arguments');
  end
  
  % Initialize algorithm parametes
  maxIters = 1000;
  k = 1;
  err = 100;
  beta = randn( size(tX, 2), 1 );
  Lold = computeCostMSE(y, tX, beta);

  % Termination of iterations
  fprintf('Starting iterations, press Ctrl+c to break\n');


  % Gradient descent iteration
  while ( k <= maxIters && err > eps )
        
    % Gradient computation
    g = computeGradient(y, tX, beta); 
      
    % Updating for value of 'beta'
    beta = beta - alpha .* g;
    
    % Error computation
    L = computeCostMSE(y, tX, beta);
    err = abs(Lold - L);
    Lold = L;
    
    k = k + 1;
    
  end

end