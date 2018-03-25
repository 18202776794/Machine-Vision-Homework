%% show_data(n)
%  显示第n组数据图像
%show_data(956)

%%
clear;clc;close all;
% A_record = load('A_record.mat');

% 可以尝试修改的参数，便于加深理解，获得更优的性能。
n_hid = 40;                 % 隐层神经元数量
n_iters = 3000;              % 迭代次数
learning_rate = 0.3;       % 学习率
mini_batch_size = 500;      % batch大小

% 初始化
if exist('page_output_immediately')
    page_output_immediately(1); 
end
more off;
model = initial_model(n_hid);  % 在该函数里可以更换权值
train = load('data/train8.mat');

train.targets(:,3001:9000)=[];  % 目前总共有10000个样本
train.inputs(:,3001:9000)=[];
% 选择不同样本数 A500/B1000/C750
% train.targets(:,801:end)=[];  % 所以暂时只有500个样本???(可留待后面做CV)
% train.inputs(:,801:end)=[];
% train2.targets(:,2001:end)=[];  % 所以暂时只有500个样本???(可留待后面做CV)
% train2.inputs(:,2001:end)=[];

training = train;
n_training_cases = size(training.inputs, 2);
if n_iters ~= 0,
    test_gradient(model, training); 
end

% 训练
theta = model_to_theta(model);  % 将模型转换成向量，便于计算。
momentum_speed = theta * 0;     % 实际梯度下降数值  ？？？
training_data_losses = [];      % 训练集上的误差
for optimization_iteration_i = 1:n_iters,
    model = theta_to_model(theta);

    training_batch_start = mod((optimization_iteration_i-1) * mini_batch_size, n_training_cases)+1;
    training_batch.inputs = training.inputs(:, training_batch_start : training_batch_start + mini_batch_size - 1);
    training_batch.targets = training.targets(:, training_batch_start : training_batch_start + mini_batch_size - 1);
    gradient = model_to_theta(d_loss_by_d_model(model, training_batch));
    
    theta = theta - learning_rate * gradient;     % 更新权值

    model = theta_to_model(theta);
    training_data_losses = [training_data_losses, loss(model, training)];
    if mod(optimization_iteration_i, round(n_iters/10)) == 0,
        fprintf('After %d optimization iterations, training data loss is %f\n', optimization_iteration_i, training_data_losses(end));
    end
end

if n_iters ~= 0,
    test_gradient(model, training);
end

model = theta_to_model(theta);
if n_iters ~= 0,
clf;
hold on;
plot(training_data_losses, 'b');
legend('training');
ylabel('loss');
xlabel('iteration number');
hold off;
end
save model.mat model;

%% 验证集分类效果
% fprintf('验证集分类效果');
% train = load('data/train5.mat');
% training.targets = train.targets(:,1:800);  
% training.inputs = train.inputs(:,1:800);
% % load('data/test.mat');
% % testing = test;
% % CV验证
% testing.targets = train.targets(:,801:1000);  
% testing.inputs = train.inputs(:,801:1000);
% 
% load('model.mat');
% datas2 = {training, testing};
% data_names = {'training', 'test'};
% for data_i = 1:2,
%     data = datas2{data_i};
%     data_name = data_names{data_i};
%     fprintf('\nThe loss on the %s data is %f\n', data_name, loss(model, data));
%     fprintf('The classification sucess rate on the %s data is %.2f%%\n', data_name, 100-100*classification_performance(model, data));
% end

%% 测试分类效果
fprintf('测试集分类效果');
train = load('data/train8.mat');
training = train;
training.targets = train.targets(:,1:1000);  
training.inputs = train.inputs(:,1:1000);
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

