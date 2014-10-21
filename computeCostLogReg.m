function cost = computeCostLogReg(y, tX, beta)
    [nSamp] = size(tX, 1);
    
    cost = 0.0;
    for k = nSamp
       if(y(k) == 1)
           cost = cost + log(sigmoid(tX(k, :) * beta));
       else
           cost = cost + log(1 - sigmoid(tX(k, :) * beta));
       end
    
       cost = -(cost / nSamp);
    end
end