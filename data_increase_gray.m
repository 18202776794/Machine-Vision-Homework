%% 亮度大于1的都置1
clear;clc;close all;
train = load('data/train_rotate2.mat');
[m,n] = size(train.inputs);

figure = train.inputs(:,1:1000);

temp = zeros(m,1000);
for i = 1:m
    for j = 1:1000
        if figure(i,j)>0
            temp(i,j) = 1;
        else
            temp(i,j)=figure(i,j);
        end
    end
end

inputs = [train.inputs,double(temp)];
targets = [train.targets,train.targets(:,1:1000)];
save train_7.mat inputs targets