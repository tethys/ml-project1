function Xexp = myExp(X,degree)
% build matrix Phi for polynomial regression of a given degree
    iter = 1;
    for j = 1:size(X,2)
        for k = 1:degree
            Xexp(:,iter + k) = exp(X(:,j)/k);
        end
        iter = iter + k;
    end
end