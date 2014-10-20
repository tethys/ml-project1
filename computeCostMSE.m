function error = computeCostMSE(y, tX, beta)
    e = y - tX * beta;
    error = e'*e/(2*length(y));
end
