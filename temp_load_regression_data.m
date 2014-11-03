function [X_train, y_train, X_test, ytest, xte] = temp_load_regression_data( th )

clear data
data = load('Rome_regression.mat');
D = size(data.X_train,2);
N = size(data.X_train,1);
setSeed(1)
idx = randperm(N);

y_train = data.y_train(idx(1:1400));
X_train = data.X_train(idx(1:1400),:);

ytest = data.y_train(idx(1201:end));
xte = data.X_train(idx(1201:end), :);

if th == 0
   % indices =  (y_train < 4900);
   indices = (y_train < 4900) & (X_train(:,36) <= -13) 
   %indices = (X_train(:,36) <= -13);
else
    indices = (y_train >= 4900) & (X_train(:,36) > -13);
    %indices = (X_train(:,36) > -13);
end
y_train = y_train(indices);
X_train = X_train(indices,:);

%% Make categorical features dummy variables
categorical_data = X_train(:,D-6:end) + 1;
temp = dummyvar(categorical_data);
X_train = [X_train(:,1:(D-7)) temp];

X_test = xte;
categorical_data = X_test(:,D-6:end) + 1;
temp = dummyvar(categorical_data);
X_test = [X_test(:,1:(D-7)) temp];

%% Compute mean and std of the training set
X_mean = mean(X_train);
X_std = std(X_train);

%% Normalize train data to have 0 mean and 1 std
N = size(X_train, 1);
X_mean_rep = repmat(X_mean,[N,1]);
X_std_rep = repmat(X_std,[N,1]);
X_train = X_train - X_mean_rep;
X_train = X_train ./ X_std_rep;

%% Do the same for the test data
X_mean_test = mean(X_test);
X_std_test = std(X_test);
N = size(X_test, 1);
X_mean_rep = repmat(X_mean_test, [N,1]);
X_std_rep = repmat(X_std_test, [N,1]);
X_test = X_test - X_mean_rep;
X_test = X_test ./ X_std_rep;


%% Remove outliers from the train data
% If the data are normally distributed with mean 0 and std 1, then the
% 99.99% of them are appear between the values -3.891 and 3.891
[row_to_remove, ~] = find(X_train < -3.891);
[temp, ~] = find(X_train > 3.891);
row_to_remove = unique([row_to_remove ; temp]);
row_to_remove = sort(row_to_remove,'descend');
X_train(row_to_remove,:)=[];
length(y_train)
y_train(row_to_remove,:)=[];
length(y_train)


% 
% for i=1:36
%    scatterhist(X_train(:,i), y_train) 
%    pause
%    close
% end
end
