function error = computeCostMSE(y, tX, beta)
% Mean Square Error computation

    e = y - tX * beta;
    error = e'*e/(length(y));

end