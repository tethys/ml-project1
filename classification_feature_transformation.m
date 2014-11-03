close all
clear all

[X_train, y_train, X_test] = load_curated_classification_data;

N = size(X_train,1);
d=3;
Xp = myPoly(X_train, d);
tX = [ones(N,1) Xp];

% alpha=10.0 LR = 94.47 and pLR = 94.067 --> best results
% alpha=0.1 LR = 93.47 and pLR = 94.20 --> best results
alpha =[0.1,1,10]; 
% but it doesn't decrease (LR and pLR) until lambda=0.1 
% while increasing lambda
lambda = [1e-3, 1e-2, 1e-1, 10, 100]; 

% d = 2, 
for d = 2:2
  
Xp = myPoly(X_train, d);
tX = [ones(N,1) Xp];
for a =1:length(alpha)
    for l = 1:length(lambda)
    % split data in K fold (we will only create indices)
    K = 5;
    setSeed(1);
    N = size(y_train,1);
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
        
        yTe = y_train(idxTe);
        tXTe = tX(idxTe,:);
        yTr = y_train(idxTr);
        tXTr = tX(idxTr,:);

        [beta1] = logisticRegression(yTr, tXTr, alpha(a));
        pHatn1 = sigmoid(tXTe*beta1);
        yHatn1 = zeros(size(tXTe,1), 1);
        yHatn1(pHatn1 >= 0.5) = 1;
        test_01Loss_LR(k) = mean(double(yHatn1 == yTe))*100;
        
        [beta2] = penLogisticRegression(yTr, tXTr, alpha(a), lambda(l));
        pHatn2 = sigmoid(tXTe*beta2);
        yHatn2 = zeros(size(tXTe,1), 1);
        yHatn2(pHatn2 >= 0.5) = 1;
        test_01Loss_pLR(k) = mean(double(yHatn2 == yTe))*100;
    end
    
    fprintf('Test Accuracy normal logistic: %f\n', mean(test_01Loss_LR));
    fprintf('Test Accuracy penalized logistic: %f\n', mean(test_01Loss_pLR));
    fprintf('Test Accuracy penalized logistic: %f\n', std(test_01Loss_pLR));
    
    boxplot( test_01Loss_pLR);
    set(gca,'xtick',1:2)
    ylabel('Test Accuracy');
    set(gca,'xticklabel',{'LogisticRegression','PenLogisticRegression'})
    
    fprintf('d, alpha, l %f %f %f\n',d, alpha(a),lambda(l))
    %pause
    %close
    end
end
end