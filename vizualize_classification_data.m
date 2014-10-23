%% author Viviana Petrescu
%% 19.10.2014
%%
%% TODO try least squares
%% Try dummy variables
%% Try PCA for 3
%% Remove outliers
%%
%%

data = load('Rome_classification.mat');
D = size(data.X_train,2);
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
for i=1:D - 1
   %scatterhist(X_train(:,i), y_train); 
   myBlue = [0.06 0.06 1];
   myRed = [1 0.06 0.06];
   males = y_train==1;
   females = y_train==-1;
   
   plot(X_train_normalised(males,i), X_train_normalised(males,i+1),'xr','color',myRed,'linewidth', ...
2, 'markerfacecolor', myRed);
hold on
plot(X_train_normalised(females,i), X_train_normalised(females,i+1),'or','color', ...
myBlue,'linewidth', 2, 'markerfacecolor', myBlue);
xlabel('height');
ylabel('weight');
grid on;
   pause
   close;
end