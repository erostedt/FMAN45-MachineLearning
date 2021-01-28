function [y, cluster_label] = K_means_classifier(X, X_label, C, K)
    
    cluster_label = zeros(K,1);
    y = step_assign_cluster(X, C);
    % I now count number of zeros and ones in each cluster, then I'm
    % looking at which is the biggest and assigning the largest value as
    % the cluster label.
    for i=1:K
        nbr_of_zeros = sum(X_label(y==i)==0); % count the nbr of zero-labeled samples in the cluster
        nbr_of_ones = sum(X_label(y==i)==1); % count the nbr of one-labeled samples in the cluster
        if nbr_of_zeros < nbr_of_ones
            cluster_label(i) = 1;
        else
            cluster_label(i) = 0;
        end
    end
    
end

function y = step_assign_cluster(X,C) 
% Calculates a label from 1:K (the nbr of clusters) for each sample
% Input:        X  All samples, 784xN
%               C  A matrix, all centroids
%
% Output:       y  A vector with labels, Nx1

    [~,N] = size(X);
    y = zeros(N,1);
    
    for i=1:N
        d = fxdist(X(:,i),C); % Calculate the distance between each sample and each cluster
        [~,I] = min(d); 
        y(i) = I; % The 
    end
end

function d = fxdist(x,C) 

% Calculates the distance (2-norm) between the sample and all the centroids
% Input:        x  A column vector, 784x1, (one sample)
%               C  A matrix, 784xN (all centroids)
%
% Output:       d  Returns a vector, 1xN

    d = [];
    [~,N] = size(C);
    for i=1:N
       d(i) = norm(x-C(:,i)); 
    end
end