function beta = logisticRegression( varargin )
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
  beta = randn( size(tX, 2), 1 );
  
  for k = 1 : maxIters

      sig = sigmoid( tX * beta );
      s = sig .* (1 - sig);
      g = computeGradientLogReg( y, tX, beta );
      inv_H = inv(tX' * diag(s) * tX);
      beta = beta - alpha .* inv_H * g;

  end
  
end