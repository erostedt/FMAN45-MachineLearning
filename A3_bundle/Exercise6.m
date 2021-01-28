%%  
mnist_starter()
%%
load('models/network_trained_with_momentum.mat')

%%
x_test = loadMNISTImages('data/mnist/t10k-images.idx3-ubyte');
y_test = loadMNISTLabels('data/mnist/t10k-labels.idx1-ubyte');
y_test(y_test==0) = 10;
x_test = reshape(x_test, [28, 28, 1, 10000]);

%% Ex 6 plot filter
filters = net.layers{1, 2}.params.weights;

for filter_idx = 1:16
    subplot(4, 4, filter_idx)
    heatmap(filters(:, :, :, filter_idx), 'ColorbarVisible', 'off');
end
%% Get misclassifications
sz = length(y_test); 
label_pred = zeros(sz, 1);

batch_size = 16;
for image=1:batch_size:sz
    end_of_interval = min(image + batch_size, sz+1);
    [res, ~] = evaluate(net, x_test(:, :, :, image: end_of_interval - 1), y_test(image: end_of_interval - 1));
    
    [~, label_pred(image: end_of_interval - 1)] = max(res{end-1});
end

misclassified_idx = find(label_pred ~= y_test);
misclassified = x_test(:, :, :, misclassified_idx);

%% Plot batch of misclassifications
figure
for i=1:batch_size
    subplot(4, 4, i)
    imshow(misclassified(:, :, :, i));
end

%% Construct confusion matrix
conf_mat = confusionmat(y_test, label_pred);
heatmap(conf_mat, 'ColorbarVisible', 'off');

%% Calculate precision and recall

tp = diag(conf_mat);
sum_rows = sum(conf_mat);
sum_cols = sum(conf_mat, 2);

precision = tp./sum_rows';
recall = tp./sum_cols;


