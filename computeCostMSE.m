function error = computeCostMSE(y, tX, beta)
% Mean Square Error computation

    e = y - tX * beta;
    error = e'*e/(2*length(y));

end