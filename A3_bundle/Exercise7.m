cifar10_starter()
%%
load('models/cifar10_addedlayers.mat')

%% plot filters
filters = net.layers{1, 2}.params.weights;
filters = rescale(filters);

for filter_idx = 1:16
    subplot(4, 4, filter_idx)
    imshow(filters(:, :, :, filter_idx));% 'ColorbarVisible', 'off');
end
%%
[x_train, y_train, x_test, y_test, classes] = load_cifar10(2);

pred = zeros(numel(y_test),1);
batch = 16;
for i=1:batch:size(y_test)
    idx = i:min(i+batch-1, numel(y_test));
    % note that y_test is only used for the loss and not the prediction
    y = evaluate(net, x_test(:,:,:,idx), y_test(idx));
    [~, p] = max(y{end-1}, [], 1);
    pred(idx) = p;
end

misclassified_idx = find(pred ~= y_test);
misclassified_imgs = x_test(:, :, :, misclassified_idx);
missclassifications = pred(misclassified_idx);

%% Plot batch of misclassifications
figure
for i=1:16
    subplot(4, 4, i)
    imshow(misclassified_imgs(:, :, :, i)./255);
    curr_idx = misclassified_idx(i);
    title([strcat('True: ', classes( y_test( curr_idx ))), strcat('Guessed: ', classes( missclassifications(i) ) )])
end

%% Construct confusion matrix
conf_mat = confusionmat(double(y_test), pred);
heatmap(conf_mat, 'ColorbarVisible', 'off');

%% Calculate precision and recall

tp = diag(conf_mat);
sum_rows = sum(conf_mat);
sum_cols = sum(conf_mat, 2);

precision = tp./sum_rows';
recall = tp./sum_cols;
