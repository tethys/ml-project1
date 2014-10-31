function beta = ridgeRegression ( varargin )

  % Chech the arguments
  switch nargin
      case 2
          y = varargin{1};
          tX = varargin{2};
          lambda = 1e-5;
      case 3
          y = varargin{1};
          tX = varargin{2};
          lambda = varargin{3};
      otherwise
          error('Unexpected number of input arguments');
  end
  
  beta = pinv(tX' * tX + lambda * eye( size(tX, 2) )) * (tX' * y);

end