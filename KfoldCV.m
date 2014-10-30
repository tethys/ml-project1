% K-fold cross validation for estimating test errors
function [meanTrainError, meanValidationError]= KfoldCV(K, X, y, mode, varargin)
% Chech the arguments

    switch nargin
    case 5
        alpha = varargin{1};
    case 6
        alpha = varargin{1};
        lambda = varargin{2};
    otherwise
        error('Unexpected number of input arguments');
    end

    % split data in K fold (we will only create indices)
    setSeed(1);
    N = size(y,1);
    idx = randperm(N);
    Nk = floor(N/K);
    for k = 1:K
        idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
    end
    trainError = zeros(K,1); 
    validationError = zeros(K, 1);
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
        if(strcmp(mode, 'leastSquaresGD') == 1)         %  leastSquaresGD
           [beta] = leastSquaresGD(yTr, tXTr, alpha);
           trainError(k) = computeCostMSE(yTr, tXTr, beta);
           validationError(k) = computeCostMSE(yTe, tXTe, beta);
        elseif(strcmp(mode, 'leastSquares') == 1)     %  leastSquares
           [beta] = leastSquares(yTr, tXTr);
           trainError(k) = computeCostMSE(yTr, tXTr, beta);
           validationError(k) = computeCostMSE(yTe, tXTe, beta);
        elseif(strcmp(mode, 'ridgeRegression') == 1)     %  ridgeRegression
           [beta] = ridgeRegression(yTr, tXTr, lambda);
           beta
           [trainError(k), ~] = computeCostGradMSERegression(yTr, tXTr, beta, lambda);
           [validationError(k), ~] = computeCostGradMSERegression(yTe, tXTe, beta, lambda);
        elseif( strcmp(mode, 'logisticRegression') == 1)     %  logisticRegression
           [beta] = logisticRegression(yTr, tXTr, alpha);
           [trainError(k), ~] = computeCostGradLogisticRegression(yTr, tXTr, beta, 0);
           [validationError(k), ~] = computeCostGradLogisticRegression(yTe, tXTe, beta, 0);
        elseif (strcmp(mode, 'penLogisticRegression') == 1)                    %  penLogisticRegression
           [beta] = penLogisticRegression(yTr, XTr, alpha, lambda);
           [trainError(k), ~] = computeCostGradLogisticRegression(yTr, tXTr, beta, lambda);
           [validationError(k), ~] = computeCostGradLogisticRegression(yTe, tXTe, beta, lambda);
        else
            error('Wrong mode')
        end 
    end
    meanTrainError = mean(trainError);
    meanValidationError = mean(validationError);
end