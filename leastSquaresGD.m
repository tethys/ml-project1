function  beta = leastSquaresGD ( varargin )
%
% 'beta' computation for the method least Squares using Gradient Descent
%
% Mandatory inputs:
% y      : y of the training dataset
% tX     : X of the training dataset
%
% Optional inputs:
% alpha : method parameter alpha
%
% Outputs:
% beta   : estimated coefficients of the model
%

  % Chech the arguments
  switch nargin
      case 2
          y = varargin{1};
          tX = varargin{2};
          alpha = 0.1;
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
  err = 1 ./ eps;
  beta = randn( size( tX, 2 ), 1 );
  Lold = computeCostRMSE( y, tX, beta );

  % Gradient descent iteration
  while ( k <= maxIters && err > eps )
        
    % Gradient computation
    g = computeGradient(y, tX, beta); 
      
    % Updating for value of 'beta'
    beta = beta - alpha .* g;
    
    % Error computation
    L = computeCostRMSE(y, tX, beta);
    err = abs(Lold - L);
    Lold = L;
    
    k = k + 1;
    
  end

end