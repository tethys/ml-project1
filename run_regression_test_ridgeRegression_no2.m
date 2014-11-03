%% Loading data - Fit parameters
% First fitting
close all
clear all
clc

lambda = 1e-7;
rmseTr_1 = zeros(17,1);
rmseVa_1 = zeros(17,1);
rmseTr_2 = zeros(17,1);
rmseVa_2 = zeros(17,1);
for j = 1 : 50
    [X_train_all, y_train_all, ~, ~] = load_regression_data(0);
    % Keep randomly 80% of the data for training
    percentage = 0.80;
    N = round(percentage * size(X_train_all,1));

    ind_train = randperm(size(X_train_all,1));
    ind_train = ind_train(1:N);
    X_train = X_train_all(ind_train,:);
    y_train = y_train_all(ind_train,1);

    ind_test = find(ismember(1:size(X_train_all,1), ind_train)==0);
    X_valid = X_train_all(ind_test,:);
    y_valid = y_train_all(ind_test,1);

    % rmseTr_1 = zeros(40,1);
    % rmseVa_1 = zeros(40,1);
    for i = 0 : 16
        percentage = 0.1 + 0.05*i;
        K = round(percentage * size(X_train,1));
        Xtr = [ones(K,1) X_train(1:K,:)];
        y_tr = y_train(1:K,1);
        betaLS_1 = ridgeRegression(y_tr, Xtr, lambda);
        rmseTr_1(i+1) = rmseTr_1(i+1) + computeCostRMSE(y_tr, Xtr, betaLS_1);
        Xva = [ones(size(y_valid)) X_valid];
        rmseVa_1(i+1) = rmseVa_1(i+1) + computeCostRMSE(y_valid, Xva, betaLS_1);
    end

    % Second fitting
    %clearvars -except lambda rmseTr_1 rmseVa_1

    [X_train_all, y_train_all, ~, ~] = load_regression_data(1);
    % Keep randomly 80% of the data for training
    percentage = 0.80;
    N = round(percentage * size(X_train_all,1));

    ind_train = randperm(size(X_train_all,1));
    ind_train = ind_train(1:N);
    X_train = X_train_all(ind_train,:);
    y_train = y_train_all(ind_train,1);

    ind_test = find(ismember(1:size(X_train_all,1), ind_train)==0);
    X_valid = X_train_all(ind_test,:);
    y_valid = y_train_all(ind_test,1);

    % rmseTr_2 = zeros(40,1);
    % rmseVa_2 = zeros(40,1);
    for i = 0 : 16
        percentage = 0.1 + 0.05*i;
        K = round(percentage * size(X_train,1));
        Xtr = [ones(K,1) X_train(1:K,:)];
        y_tr = y_train(1:K,1);
        betaLS_2 = ridgeRegression(y_tr, Xtr, lambda);
        rmseTr_2(i+1) = rmseTr_2(i+1) + computeCostRMSE(y_tr, Xtr, betaLS_2);
        Xva = [ones(size(y_valid)) X_valid];
        rmseVa_2(i+1) = rmseVa_2(i+1) + computeCostRMSE(y_valid, Xva, betaLS_2);
    end
end

rmseTr_1 = rmseTr_1 ./ 50;
rmseVa_1 = rmseVa_1 ./ 50;
rmseTr_2 = rmseTr_2 ./ 50;
rmseVa_2 = rmseVa_2 ./ 50;

%% Plotting
percentage = 0.10:0.05:0.9;
figure;
subplot(2,1,1);
plot(percentage,rmseTr_1, 'b', 'LineWidth', 2);
hold on;
plot(percentage,rmseVa_1, 'r', 'LineWidth', 2);
hx = xlabel('Percentage of the training data');
hy = ylabel('RMSE');
set(gca,'fontsize',14,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',14,'fontname','avantgarde','color',[.3 .3 .3]);
xlim([0.1 0.9]);
ylim([50 250]);
grid on;

subplot(2,1,2);
plot(percentage,rmseTr_2, 'b', 'LineWidth', 2);
hold on;
plot(percentage,rmseVa_2, 'r', 'LineWidth', 2);
hx = xlabel('Percentage of the training data');
hy = ylabel('RMSE');
set(gca,'fontsize',14,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',14,'fontname','avantgarde','color',[.3 .3 .3]);
xlim([0.1 0.9]);
ylim([0 350]);
grid on;

print -dpdf learning_curve.pdf;
