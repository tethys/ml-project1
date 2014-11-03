function  error = computeCostRMSE ( y, tX, beta )
%
% RMSE computation given the model coefficients and the data
%
% Mandatory inputs:
% y      : y of the training dataset
% tX     : X of the training dataset
% beta   : estimated coefficients of the model
%
% Outputs:
% error  : estimated RMSE
%

  e = y - tX * beta;
  error = sqrt(e'*e./(length(y)));

end