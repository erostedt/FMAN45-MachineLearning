%% Choose params
close all
clear
load 'A2_data.mat'

d = 2;
K = 2;

X_train = train_data_01;
X_train_label = train_labels_01;

X_test = test_data_01;
X_test_label = test_labels_01;

%% E1, ALWAYS RUN THIS ONE BEFORE RUNNING E2-E4


% Preping the data
[D,N] = size(X_train);
P = min(D,N);

% Normalizing X so that the mean is 0 for each column
X_norm = X_train - mean(X_train,2);

% Calculating U, the left singular vectors
%[U,eig_val] = eig(X_norm*X_norm');
[U,~,~] = svd(X_norm);

% Calculating the reduced matrix
U_new = [];
for i=1:d
    U_new = [U_new, U(:,i)];
end
X_new = X_norm'*U_new; % projecting on the principal components

% Ploting
figure(1);
p = plot(X_new(X_train_label==0,1), X_new(X_train_label==0,2),'ro',...,
    X_new(X_train_label==1,1),X_new(X_train_label==1,2),'b+');
p(1).MarkerSize = 3;
p(2).MarkerSize = 3;
xlabel('First principal component');
ylabel('Second principal component');
lgd = legend('Class 1','Class 2');
lgd.FontSize = 15;
title 'PCA';
set(gca, 'FontSize', 13);

%% E2 plot

[idxs, centroid] = K_means_clustering(X_train,K);

centroid_plus_mean = centroid;

for i=1:K
   centroid(:,i) = centroid(:,i)-mean(X_train,2); 
end

projected_centroid = U_new'*centroid; % Projecting the centroids onto the principal components

if K==2
    % Ploting the figure
    figure(1);
    p = plot(X_new(idxs==1,1), X_new(idxs==1,2),'ro',...,
        X_new(idxs==2,1),X_new(idxs==2,2),'b+',projected_centroid(1,:),projected_centroid(2,:),...,
        'kx');
    p(1).MarkerSize = 3;
    p(2).MarkerSize = 3;
    p(3).MarkerSize = 10;
    xlabel('First principal component');
    ylabel('Second principal component');
    lgd = legend('Cluster 1','Cluster 2','Centroids');
    lgd.FontSize = 15;
    title 'Cluster Assignments';
    set(gca, 'FontSize', 13);
elseif K==5
    % Ploting the figure
    figure(1);
    p = plot(X_new(idxs==1,1), X_new(idxs==1,2),'ro',...,
        X_new(idxs==2,1),X_new(idxs==2,2),'b+',...,
        X_new(idxs==3,1), X_new(idxs==3,2),'g*',...,
        X_new(idxs==4,1), X_new(idxs==4,2),'c.',...,
        X_new(idxs==5,1), X_new(idxs==5,2),'ms',...,
        projected_centroid(1,:),projected_centroid(2,:),'kx');
    p(1).MarkerSize = 3;
    p(2).MarkerSize = 3;
    p(3).MarkerSize = 3;
    p(4).MarkerSize = 3;
    p(5).MarkerSize = 3;
    p(6).MarkerSize = 10;
    xlabel('First principal component');
    ylabel('Second principal component');
    lgd = legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5','Centroids');
    lgd.FontSize = 15;
    title 'Cluster Assignments';
    set(gca, 'FontSize', 13);
end

%% E3
if K==2
    figure(1)

    subplot(1,2,1)
    imshow(reshape(centroid_plus_mean(:,1),28,28))
    title 'Cluster 1';
    set(gca, 'FontSize', 13);
    subplot(1,2,2)
    imshow(reshape(centroid_plus_mean(:,2),28,28))
    title 'Cluster 2';
    set(gca, 'FontSize', 13);
    
elseif K==5

    % Ploting
    figure(1)

    subplot(1,5,1)
    imshow(reshape(centroid_plus_mean(:,1),28,28))
    title 'Cluster 1';
    set(gca, 'FontSize', 12);

    subplot(1,5,2)
    imshow(reshape(centroid_plus_mean(:,2),28,28))
    title 'Cluster 2';
    set(gca, 'FontSize', 12);

    subplot(1,5,3)
    imshow(reshape(centroid_plus_mean(:,3),28,28))
    title 'Cluster 3';
    set(gca, 'FontSize', 12);

    subplot(1,5,4)
    imshow(reshape(centroid_plus_mean(:,4),28,28))
    title 'Cluster 4';
    set(gca, 'FontSize', 12);

    subplot(1,5,5)
    imshow(reshape(centroid_plus_mean(:,5),28,28))
    title 'Cluster 5';
    set(gca, 'FontSize', 12);
end

%% E4 a)
[y, cluster_label] = K_means_classifier(X_test, X_test_label, centroid, K);

%% E4 b) + E5

% The data needed for the table
label_one = zeros(K,1);
label_zero = zeros(K,1);
misclassified = zeros(K,1);
[~,N] = size(X_test); % N_train / N_test

for i=1:K
    [size_of_cluster,~] = size(find(y==i));
    
    nbr_of_zeros = sum(X_test_label(y==i)==0); % count the nbr of zero-labeled samples in the cluster
    nbr_of_ones = sum(X_test_label(y==i)==1); % count the nbr of one-labeled samples in the cluster
    
    label_zero(i) = nbr_of_zeros;
    label_one(i) = nbr_of_ones;
    
    if cluster_label(i)==1
        misclassified(i) = nbr_of_zeros;
    else
        misclassified(i) = nbr_of_ones;
    end
end

Sum_misclassified = sum(misclassified);
Misclassification_rate = Sum_misclassified/N*100; % The misclassification rate in percentage

%% E6
close all
clear all
load 'A2_data.mat'

% Train data
X_train = train_data_01';
T_train = train_labels_01;

% Test data
X_test = test_data_01';
T_test = test_labels_01;

model = fitcsvm(X_train, T_train);
predicted_labels = predict(model,X_test);

% This is the matrix from the A2 description
classification_matrix = zeros(2,2);

[N,~] = size(X_test); % N_train / N_test

% Top row
nbr_true_zero_pred = sum(T_test(predicted_labels==0)==0);
nbr_false_zero_pred = sum(T_test(predicted_labels==0)==1);

%Bottom row
nbr_false_one_pred = sum(T_test(predicted_labels==1)==0);
nbr_true_one_pred = sum(T_test(predicted_labels==1)==1);

% Filling in my matrix
classification_matrix(1,1) = nbr_true_zero_pred;
classification_matrix(1,2) = nbr_false_zero_pred;
classification_matrix(2,1) = nbr_false_one_pred;
classification_matrix(2,2) = nbr_true_one_pred;

sum_misclassified = classification_matrix(1,2) + classification_matrix(2,1);
Misclassification_rate = 100*sum_misclassified/N;

%% E7
close all
clear all
load 'A2_data.mat'

% Train data
X_train = train_data_01';
T_train = train_labels_01;

% Test data
X_test = test_data_01';
T_test = test_labels_01;

model = fitcsvm(X_train, T_train,'KernelFunction','gaussian', 'KernelScale', 5);

predicted_labels = predict(model,X_test);

% This is the matrix from the A2 description
classification_matrix = zeros(2,2);

[N,~] = size(X_test); % N_train / N_test

% Top row
nbr_true_zero_pred = sum(T_test(predicted_labels==0)==0);
nbr_false_zero_pred = sum(T_test(predicted_labels==0)==1);

%Bottom row
nbr_false_one_pred = sum(T_test(predicted_labels==1)==0);
nbr_true_one_pred = sum(T_test(predicted_labels==1)==1);

% Filling in my matrix
classification_matrix(1,1) = nbr_true_zero_pred;
classification_matrix(1,2) = nbr_false_zero_pred;
classification_matrix(2,1) = nbr_false_one_pred;
classification_matrix(2,2) = nbr_true_one_pred;

sum_misclassified = classification_matrix(1,2) + classification_matrix(2,1);
Misclassification_rate = 100*sum_misclassified/N;
