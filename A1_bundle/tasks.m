load A1_data.mat

%% Calculations and plot for task 4
what = lasso_ccd(t, X, 2, zeros(1000,1));

title('Reconstruction plot for \lambda=2', 'FontSize', 18)
xlabel('Time', 'FontSize', 14) 
ylabel('Noisy data', 'FontSize', 14)
hold on

% Original data
scatter(n, t, 20 ,'r', 'x');

% Regression
y = X*what;
scatter(n, y, 20, 'b');

% Interpolation
plot(ninterp, Xinterp*what, 'g')
legend({'Original data','Prediction', 'Interpolation'}, 'FontSize', 10)

%% Calculate how many non-negative weights are needed
non_zero = sum(what ~= 0);


%% Task 5 - Lasso-CV

lambda_max = max(abs(X'*t));
lambda_min = 0.5;
N_lambda = 50;
lambdavec = exp(linspace(log(lambda_min), log(lambda_max), N_lambda));
[wopt,lambdaopt,RMSEval,RMSEest] = lasso_cv(t, X, lambdavec, 10);
%% Plot
title('RMSE for different lambdas', 'FontSize', 18)
xlabel('Lambda', 'FontSize', 14) 
ylabel('Error', 'FontSize', 14)
hold on

plot(lambdavec, RMSEval, 'x');
plot(lambdavec, RMSEest, 'o');
plot([lambdaopt lambdaopt], [0, 4], '--');
xlim([lambda_min,lambda_max])
set(gca, 'XTick', unique([lambdaopt, get(gca, 'XTick')]));
set(gca,'FontSize',18)

legend({'RMSE for validation','RMSE for estimation', 'Optimal lambda'}, 'FontSize', 10)

%% Reconstruction plot with the optimal lambda
title('Reconstruction plot for \lambda=1.9312', 'FontSize', 18)
xlabel('Time', 'FontSize', 14) 
ylabel('Noisy data', 'FontSize', 14)
hold on

% Original data
scatter(n, t, 20 ,'r', 'x');

% Regression
y = X*wopt;
scatter(n, y, 20, 'b');

% Interpolation
plot(ninterp, Xinterp*wopt, 'g')
legend({'Original data','Best prediction', 'Interpolation'}, 'FontSize', 10)

%% Task 6 - Calculate Multiframe Lasso Cross Validation for different lambdas
lambda_max = 0.03; % max(abs(Xaudio'*Ttrain));
lambda_min = 0.0001;
N_lambda = 50;
lambdavec = exp(linspace(log(lambda_min), log(lambda_max), N_lambda));
[wopt,lambdaopt,RMSEval,RMSEest] = multiframe_lasso_cv(Ttrain, Xaudio, lambdavec, 10);

%% Plot the RMSE for the different lambdas
title('RMSE for different lambdas', 'FontSize', 18)
xlabel('Lambda', 'FontSize', 14) 
ylabel('Error', 'FontSize', 14)
hold on

plot(lambdavec, RMSEval, 'x');
plot(lambdavec, RMSEest, 'o');
plot([lambdaopt lambdaopt], [0, 0.15], '--');
xlim([lambda_min,lambda_max])

legend({'RMSE for validation','RMSE for estimation', 'Optimal \lambda = 0.0047'}, 'FontSize', 10)

%% Task 7 -  Denoise the test audio
Y_clean = lasso_denoise(Ttest, Xaudio, 0.0047);

%% Save the denoised audio
save('denoised_audio','Y_clean','fs')