%% 测试分类效果
load('data/train.mat');
training = train;
% training.targets = train.targets(:,1:750);  
% training.inputs = train.inputs(:,1:750);
load('data/test.mat');
testing = test;
% CV验证
% testing.targets = train.targets(:,751:1000);  
% testing.inputs = train.inputs(:,751:1000);

load('model.mat');
datas2 = {training, testing};
data_names = {'training', 'test'};
for data_i = 1:2,
    data = datas2{data_i};
    data_name = data_names{data_i};
    fprintf('\nThe loss on the %s data is %f\n', data_name, loss(model, data));
    fprintf('The classification sucess rate on the %s data is %.2f%%\n', data_name, 100-100*classification_performance(model, data));
end