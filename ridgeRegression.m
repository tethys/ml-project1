function  beta = ridgeRegression ( varargin )
%
% 'beta' computation for the method Ridge Regression
%
% Mandatory inputs:
% y      : y of the training dataset
% tX     : X of the training dataset
%
% Optional inputs:
% lambda : method parameter lambda
%
% Outputs:
% beta   : estimated coefficients of the model
%

  % Chech the arguments
  switch nargin
      case 2
          y = varargin{1};
          tX = varargin{2};
          lambda = 1e-7;
      case 3
          y = varargin{1};
          tX = varargin{2};
          lambda = varargin{3};
      otherwise
          error('Unexpected number of input arguments');
  end
  
  % Compute the coefficients
  beta = (tX' * tX + lambda * eye( size(tX, 2) )) \ (tX' * y);

end