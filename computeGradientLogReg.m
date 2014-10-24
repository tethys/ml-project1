function g = computeGradientLogReg(y, tX, beta)

	nSamp = size(tX, 1);
    g = tX' * ((sigmoid(tX * beta)) - y);

end