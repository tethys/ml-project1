% Logistic Regression
  clear all

 % Load data
  load('height_weight_gender.mat');
  height = height * 0.025;
  weight = weight * 0.454; 

  y = gender;
  X = [height(:) weight(:)];

 % randomly permute data
  N = length(y);
  idx = randperm(N);

  y = y(idx);
  X = X(idx,:);

  % subsample
  y = y(1:200);
  X = X(1:200,:);
  
  N = length(y);
  tX = [ones(N,1) X];

  % algorithm parametes
  maxIters = 1000;
  alpha = 0.01;
  converged = 0;

  % initialize
  % beta = [0; 0; 0];
  beta = zeros(size(tX,2), 1);
  pL = 0.0;
  
  % visualize the data
  %x1 = X(:, 1);
  %x2 = X(:, 2);
  %figure; hold on;
  %h1 = scatter(x1(y==0),x2(y==0),50,'b','filled');  % blue dots for 0
  %h2 = scatter(x1(y==1),x2(y==1),50,'r','filled');  % red dots for 1
  %set([h1 h2],'MarkerEdgeColor',[.5 .5 .5]);        % outline dots in gray
  %xlabel('height');
  %ylabel('weight');
  
  % iterate
  fprintf('Starting iterations, press Ctrl+c to break\n');
  fprintf('L  beta0 beta1\n'); 

  for k = 1:maxIters
    % INSERT YOUR FUNCTION FOR COMPUTING GRADIENT 
    %g = computeGradient(y, tX, beta);
    g = computeGradientLogReg(y, tX, beta);
    
    % INSERT YOUR FUNCTION FOR COMPUTING COST FUNCTION
    %L = computeCost(y, tX, beta);
    L = computeCostLogReg(y, tX, beta);

    % INSERT GRADIENT DESCENT UPDATE TO FIND BETA
    beta = beta - alpha .* g;
    
    % INSERT CODE FOR CONVERGENCE
    if (abs(L - pL) \ L) <= 0.0000001
      break;
    end
    
    pL = L;

    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;

    % print
    fprintf('%.6f  %.4f %.4f &.4f\n', L, beta(1), beta(2), beta(3));
  end
  
figure; hold on;
x1 = X(:, 1);
x2 = X(:, 2);
scatter(x1(y==0),x2(y==0),50,'b','filled');  
scatter(x1(y==1),x2(y==1),50,'r','filled'); 
xlabel('height');
ylabel('weight');

% a grid of points to evaluate the model
ax = axis; 
h = linspace(ax(1),ax(2),100);
w = linspace(ax(3),ax(4),100);
[hx,wx] = meshgrid(h,w);

K = [hx(:) wx(:)];
tK = [ones(length(K(:,1)), 1) K];
 
pred = reshape(sigmoid(tK*beta),[length(h) length(w)]); 

%plot the decision surface
% (draws contour plots of pred Matrix. hx and hy specify the x- and y-axis
% limits)
contour(hx,wx,pred, 1);
hold on;

% plot indiviual data points
% x1 = X(:, 1);
% x2 = X(:, 2);
% myBlue = [0.06 0.06 1];
% myRed = [1 0.06 0.06];
% plot(x1(y==0),x2(y==0),'xr','color',myRed,'linewidth', 2, 'markerfacecolor', myRed);
% plot(x1(y==1),x2(y==1),'or','color', myBlue,'linewidth', 2, 'markerfacecolor', myBlue);
% xlabel('height');
% ylabel('weight');
% %xlim([min(h) max(h)]);
% %ylim([min(w) max(w)]);
% grid on;
  
  


