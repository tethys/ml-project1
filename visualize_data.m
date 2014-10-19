%% author Viviana Petrescu
%% 19.10.2014
%%
%% TODO try least squares
%% Try dummy variables
%% Try PCA for 3
%% Remove outliers
%%
%%

data = load('Rome_regression.mat');
D = size(data.X_train,2) - 6;
X_train = data.X_train(:,1:D);
y_train = data.y_train(:,1);

%% Compute mean and std of the training set
X_mean = mean(X_train);
X_std = std(X_train);
%% plot error bar for every dimenstion
errorbar(X_mean, X_std)
pause

%% Normalize the data to have 0 mean and 1 std
N = size(X_train,1);
X_mean_rep = repmat(X_mean,[N, 1]);
X_std_rep = repmat(X_std,[N,1]);
X_train_normalised = X_train - X_mean_rep;
X_train_normalised = X_train_normalised ./ X_std_rep;

D = size(X_train,2);
for i=1:D
   scatterhist(X_train_normalised(:,i), y_train); 
  
   %pause
   %close;
end


%% SVD
%[U, S, V] = svd(X_train_normalised);
%diag(S)


