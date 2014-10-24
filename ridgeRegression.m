function beta = ridgeRegression(y, tX, lambda)

    beta = (tX' * tX + lambda * eye( size(tX, 2) )) \ (tX' * y);

end