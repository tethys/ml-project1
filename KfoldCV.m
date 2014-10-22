% K-fold cross validation for estimating test errors
function tError = KfoldCV(K, X, y, mode)

    % split data in K fold (we will only create indices)
    setSeed(1);
    K = 4;
    N = size(y,1);
    idx = randperm(N);
    Nk = floor(N/K);
    for k = 1:K
        idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
    end

    for k = 1:K
        % get k'th subgroup in test, others in train
        idxTe = idxCV(k,:);
        idxTr = idxCV([1:k-1 k+1:end],:);
        idxTr = idxTr(:);
        
        yTe = y(idxTe);
        XTe = X(idxTe,:);
        yTr = y(idxTr);
        XTr = X(idxTr,:);

        % form tX
        tXTr = [ones(length(yTr), 1) XTr];
        tXTe = [ones(length(yTe), 1) XTe];

        % calculate parameters
        if( mode == 0 )         %  leastSquaresGD
            alpha = 0.1;
           [beta] = leastSquaresGD(yTr, tXTr, alpha);
        elseif( mode == 1 )     %  leastSquares
           [beta] = leastSquares(yTr, tXTr);
        elseif( mode == 2 )     %  ridgeRegression
            lambda = 1.0;
           [beta] = ridgeRegression(yTr, tXTr, lambda);
        elseif( mode == 3 )     %  logisticRegression
            alpha = 0.1;
           [beta] = logisticRegression(yTr, tXTr, alpha);
        else                    %  penLogisticRegression
            alpha = 0.1;
            lambda = 1.0;
           [beta] = penLogisticRegression(yTr, XTr, alpha, lambda);
        end
        
        % testing MSE using least squares
        tErrorSub(k) = computeCost(yTe, tXTe, beta);
    end
    tError = mean(tErrorSub);
end