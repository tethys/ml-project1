function g = computeGradientLogReg(y, tX, beta)

    N = size( tX, 1 );
	g = tX' * (sigmoid( tX * beta ) - y);
    g = g ./ N;

end