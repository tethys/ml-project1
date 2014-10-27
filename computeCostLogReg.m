function L = computeCostLogReg(y, tX, beta)
    N = size( tX, 1 );
    L = 0.0;
    
    for i = 1 : N
        L = L - y(i) * log( sigmoid( tX(i,:) * beta ) ) - (1 - y(i)) * log( 1 - sigmoid( tX(i,:) * beta ) );
    end
    
    L = L ./ N;
    
end