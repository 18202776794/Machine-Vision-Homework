%% 利用放大缩小来增加训练集
clear;clc;close all;
train = load('train_rotate.mat');
[m,n] = size(train.inputs);

temp = zeros(m,n);
for i = 1:n
    n_figure0 = train.inputs(:,i);
    n_figure1 = reshape(n_figure0,16,16);
    n_figure2 = imresize(n_figure1,[14,14]);
    n_figure3 = zeros(16,16);
    for j = 2:15
        n_figure3(2:15,j) = n_figure2(:,j-1);
    end
    n_figure3(1,:) = zeros(1,16);
    n_figure3(16,:) = zeros(1,16);
    n_figure3(2:15,1) = zeros(14,1);
    n_figure3(2:15,16) = zeros(14,1);
    temp(:,i) = reshape(n_figure3,256,1); 
end
inputs = [train.inputs,double(temp)];
targets = [train.targets,train.targets];
save train5.mat inputs targets