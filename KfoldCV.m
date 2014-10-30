% K-fold cross validation for estimating test errors
function [meanTrainError, meanValidationError]= KfoldCV(K, tX, y, mode, varargin)
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
        tXTe = tX(idxTe,:);
        yTr = y(idxTr);
        tXTr = tX(idxTr,:);

        % calculate parameters
        if(strcmp(mode, 'leastSquaresGD') == 1)         %  leastSquaresGD
           [beta] = leastSquaresGD(yTr, tXTr, alpha);
           trainError(k) = computeCostRMSE(yTr, tXTr, beta);
           validationError(k) = computeCostRMSE(yTe, tXTe, beta);
        elseif(strcmp(mode, 'leastSquares') == 1)     %  leastSquares
           [beta] = leastSquares(yTr, tXTr);
           trainError(k) = computeCostRMSE(yTr, tXTr, beta);
           validationError(k) = computeCostRMSE(yTe, tXTe, beta);
        elseif(strcmp(mode, 'ridgeRegression') == 1)     %  ridgeRegression
           [beta] = ridgeRegression(yTr, tXTr, lambda);
           [trainError(k), ~] = computeRMSECostGradRegression(yTr, tXTr, beta, lambda);
           [validationError(k), ~] = computeRMSECostGradRegression(yTe, tXTe, beta, lambda);
        elseif( strcmp(mode, 'logisticRegression') == 1)     %  logisticRegression
           [beta] = logisticRegression(yTr, tXTr, alpha);
           [trainError(k), ~] = computeCostGradLogisticRegression(yTr, tXTr, beta, lambda);
           [validationError(k), ~] = computeCostGradLogisticRegression(yTe, tXTe, beta, lambda);
%            
%            predicted_probability = zeros(length(yTr), 1);
%            predicted_probability(sigmoid(tXTr*beta) >= 0.5) = 1;
%            trainError(k) = mean(double(predicted_probability ~= yTr)) * 100;
% 
%            predicted_probability = zeros(length(yTe), 1);
%            predicted_probability(sigmoid(tXTe*beta) >= 0.5) = 1;
%            validationError(k) = mean(double(predicted_probability ~= yTe)) * 100;
           
        elseif (strcmp(mode, 'penLogisticRegression') == 1)                    %  penLogisticRegression
           [beta] = penLogisticRegression(yTr, tXTr, alpha, lambda);
           [trainError(k), ~] = computeCostGradLogisticRegression(yTr, tXTr, beta, lambda);
           [validationError(k), ~] = computeCostGradLogisticRegression(yTe, tXTe, beta, lambda);
          % predicted_probability = zeros(length(yTr), 1);
          % predicted_probability(sigmoid(tXTr*beta) >= 0.5) = 1;
          % trainError(k) = mean(double(predicted_probability ~= yTr)) * 100;

          % predicted_probability = zeros(length(yTe), 1);
          % predicted_probability(sigmoid(tXTe*beta) >= 0.5) = 1;
          % validationError(k) = mean(double(predicted_probability ~= yTe)) * 100;
        else
            error('Wrong mode')
        end 
    end
    meanTrainError = mean(trainError);
    meanValidationError = mean(validationError);
end