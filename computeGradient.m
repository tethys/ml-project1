function  g = computeGradient ( y, tX, beta )
%
% Gradient computation given the model coefficients and the data
%
% Mandatory inputs:
% y      : y of the training dataset
% tX     : X of the training dataset
% beta   : estimated coefficients of the model
%
% Outputs:
% g      : the resulting gradient
%

  e = y - tX * beta;
  g = ( -1 / length( y ) ) * tX' * e;

end