function error = computeCostRMSE(y, tX, beta)
% Root Mean Square Error computation

    e = y - tX * beta;
    error = e'*e/(length(y));

end