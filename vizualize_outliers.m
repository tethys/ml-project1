
close all;
clear all;
data = load('Rome_regression.mat');

y_train = data.y_train(:,1);

D = size(data.X_train,2) - 7;
X_train_real = data.X_train(:,1:D);


%% Make categorical data into dummy variables
X_categorical = data.X_train(:, D+1:end) + 1;
X_dummy_var = dummyvar(X_categorical);
X_train = [X_train_real X_dummy_var];


%% Compute mean and std of the training set
X_mean = mean(X_train);
X_std = std(X_train);
%% Normalize the data to have 0 mean and 1 std
N = size(X_train,1);
X_train_normalised = X_train;


figure;
scatterhist(X_train_normalised(:,36), y_train);
hy = ylabel('y');
hx = xlabel('Feature 36 for training data');
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
hold on;
plot([-20,9],[4900,4900],'LineWidth',2, 'Color','r')
plot([-13,-13],[0,7000],'LineWidth',2, 'Color','k')

sum(X_train_normalised(:,36) > 1.4)