function  beta = leastSquares ( y, tX )
%
% 'beta' computation for the method least Squares using normal equations
%
% Inputs:
% y      : y of the training dataset
% tX     : X of the training dataset
%
% Outputs:
% beta   : estimated coefficients of the model
%

  beta = pinv(tX' * tX) * (tX' * y);

end