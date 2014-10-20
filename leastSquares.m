function beta = leastSquares(y, tX)
    beta = pinv(tX'*tX)*(tX'*y);
end