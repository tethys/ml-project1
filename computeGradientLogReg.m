function g = computeGradientLogReg(y, tX, beta)
	[nSamp] = size(tX, 1);
    g = 1./nSamp * tX' * ((sigmoid(tX * beta)) - y);
end