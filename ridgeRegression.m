function beta = ridgeRegression(y, tX, lambda)
    beta = pinv(tX'*tX + lambda*eye(size(tX, 2)))* (tX' * y);
end