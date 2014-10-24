function g = computeGradient(y, tX, beta)
% Computation of the gradient

    error = y - tX * beta;
    g = -1/length(y) * tX'*error;

end