
clear all
data = load('Rome_classification.mat');
D = size(data.X_train,2);

hist(data.y_train)
title('Distribution of y training values')
%% We observe that there are twice as many samples from one class type
%% than the other

y_train = data.y_train;
X_train = data.X_train;
%% We look at the distributino of both training and testing data to
%% see if they are similar
for i=1:D
    [N1, yout] = hist(X_train(:,i)); 
    hold on;
    N2= hist(data.X_test(:,i), yout);
    
    
    bar(yout, [N1;N2]')
  % pause;
    close
 end;
 
X_mean = mean(X_train);
X_std = std(X_train);
errorbar(X_mean, X_std)

figure
boxplot(X_train)
hx = xlabel('Mean and standard deviation of training data features');
hy = ylabel('');
% the following code makes the plot look nice and increase font size etc.
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set(gca, 'XTickLabel', [])
set([hx; hy],'fontsize',20,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
print -dpdf ClassificationDistr.pdf

%% Normalize the data to have 0 mean and 1 std
N = size(X_train,1);
X_mean_rep = repmat(X_mean,[N, 1]);
X_std_rep = repmat(X_std,[N,1]);
X_train_normalised = X_train - X_mean_rep;
X_train_normalised = X_train_normalised ./ X_std_rep;

%clustergram(X_train_normalised');
 figure;
 D = size(X_train,2);
 for i=1:D
    scatterhist(X_train_normalised(:,i), y_train); 
    pause;
    close
 end
 