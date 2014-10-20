function g = computeGradient(y, tX, beta)
    error = y - tX * beta;
    g = -1/length(y) * tX'*error;
end