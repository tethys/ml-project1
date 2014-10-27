function g = computeGradientPenLogReg( y, tX, beta, lambda )

    N = size( tX, 1 );
    d = length( beta );
    g = zeros( d, 1 );
    
	g(1) = tX(:, 1)' * (sigmoid( tX * beta ) - y) ./ N;
    for i = 2 : d   
        g(i) = tX(:, i)' * (sigmoid( tX * beta ) - y) ./ N + (lambda * beta(i) ) ./ N;
    end

end