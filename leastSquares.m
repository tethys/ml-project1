function beta = weightedLeastSquares( y, tX )
% 'beta' computation using normal equations

    beta = (tX' * tX) \ (tX' * y);

end