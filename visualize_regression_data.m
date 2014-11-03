%% author Viviana Petrescu
%% 19.10.2014
%%
%% TODO try least squares
%% Try dummy variables
%%
%%

close all;
clear all;
data = load('Rome_regression.mat');

y_train = data.y_train(:,1);

D = size(data.X_train,2) - 7;
X_train_real = data.X_train(:,1:D);
boxplot(X_train_real)
hx = xlabel('Mean and standard deviation of training data features');
hy = ylabel('');
% the following code makes the plot look nice and increase font size etc.
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set(gca, 'XTickLabel', [])
set([hx; hy],'fontsize',20,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
print -dpdf NAME.pdf

pause

%% Make categorical data into dummy variables
X_categorical = data.X_train(:, D+1:end) + 1;
X_dummy_var = dummyvar(X_categorical);
X_train = [X_train_real X_dummy_var];


%% Compute mean and std of the training set
X_mean = mean(X_train);
X_std = std(X_train);
%% Normalize the data to have 0 mean and 1 std
N = size(X_train,1);
X_mean_rep = repmat(X_mean,[N, 1]);
X_std_rep = repmat(X_std,[N,1]);
X_train_normalised = X_train - X_mean_rep;
X_train_normalised = X_train_normalised ./ X_std_rep;

%% Plot y related to every input
% figure;
 for i=36:36
%     subplot(6,6,i)
     scatterhist(X_train_normalised(:,i), y_train); 
     pause
     close
 end
% pause

%% Plot y related to every input
% figure;
% for i=1:D
%     subplot(6,6,i)
%     scatter(X_train_normalised(:,i), y_train); 
% end
% pause
 

%% plot X related to other vars
% 
% for f = 1:35
%     figure
%     fprintf(1,'Feature %d\n', f);
%     for i=f+1:D
%         subplot(6,6, i)
%         scatter(X_train_normalised(:,i), X_train_normalised(:,f));
%     end
%     pause
% end
figure
scatter(X_train_normalised(:,5), X_train_normalised(:,12));
pause
close
figure
scatter(X_train_normalised(:,17), X_train_normalised(:,20));
close
figure
scatter(X_train_normalised(:,13), X_train_normalised(:,16));
pause
close
scatter(X_train_normalised(:,2), X_train_normalised(:,24));
% 
%  
% pause
% X_categorical = data.X_train(:, D:D+6) + 1;
% X_dummy_var = dummyvar(X_categorical);
% figure
% imagesc(X_dummy_var);
% %clustergram(X_dummy_var')
% 
% 
% %% Compute mean and std of the training set
% X_mean = mean(X_train);
% X_std = std(X_train);
% %% plot error bar for every dimenstion
% errorbar(X_mean, X_std)
% 
% %% Normalize the data to have 0 mean and 1 std
% N = size(X_dummy_var,1);
% X_mean = mean(X_dummy_var);
% X_std = std(X_dummy_var);
% X_mean_rep = repmat(X_mean,[N, 1]);
% X_std_rep = repmat(X_std,[N,1]);
% X_dummy_normalised = X_dummy_var - X_mean_rep;
% X_dummy_normalised = X_dummy_normalised ./ X_std_rep;
% 
%  D = size(X_dummy_normalised,2);
%  for i=1:D
%     scatterhist(X_dummy_normalised(:,i), y_train); 
%     pause;
%     close
%  end
