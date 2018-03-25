%% data_add: �뷽�跨����������
%% �������������ݼ���ԭ�����ݼ�����ƴ��
clear;clc;close all;
load('data/train.mat');
train_add = load('data/train_add.mat');
[m0,n0] = size(train.inputs);
[m,n] = size(train_add.train_x);
temp = zeros(m0,m);
for i = 1:m
    n_figure1 = reshape(train_add.train_x(i,:),28,28);
    n_figure2 = imresize(n_figure1,[16,16]);
    n_figure3 = reshape( n_figure2,256,1);
    temp(:,i) = n_figure3;
end
train2.inputs = [train.inputs,double(temp)];
train2.targets = [train.targets,double(train_add.train_y)'];

%% ����ͼ��ƽ��������ѵ����
clear;clc;close all;
load('data/train.mat');
[m,n] = size(train.inputs);

temp = zeros(m,n);
for i = 1:n
    n_figure0 = train.inputs(:,i);
    n_figure1 = reshape(n_figure0,16,16);
    n_figure2 = myimmove(n_figure1,[1,1]);
    temp(:,i) = reshape(n_figure2,256,1); 
end
inputs = [train.inputs,double(temp)];
targets = [train.targets,train.targets];
save train3.mat inputs targets
%% ����ͼ����ת������ѵ�������ӵ�5000�飬������

clear;clc;close all;
load('data/train.mat');

[m,n] = size(train.inputs);

temp1 = zeros(m,2*n);
temp2 = zeros(m,2*n);
temp3 = zeros(m,2*n);
temp4 = zeros(m,2*n);
for i = 1:n
    n_figure0 = train.inputs(:,i);
    n_figure1 = reshape(n_figure0,16,16);
    n_figure2 = imrotate(n_figure1,1,'crop');% ��ת1��
    n_figure3 = imrotate(n_figure1,-1,'crop');% ��ת-1��
    temp1(:,i) = reshape(n_figure2,256,1);
    temp1(:,i+n) = reshape(n_figure3,256,1);      
end

for i = 1:n
    n_figure0 = train.inputs(:,i);
    n_figure1 = reshape(n_figure0,16,16);
    n_figure2 = imrotate(n_figure1,2,'crop');% ��ת2��
    n_figure3 = imrotate(n_figure1,-2,'crop');% ��ת-2��
    temp2(:,i) = reshape(n_figure2,256,1);
    temp2(:,i+n) = reshape(n_figure3,256,1);      
end

for i = 1:n
    n_figure0 = train.inputs(:,i);
    n_figure1 = reshape(n_figure0,16,16);
    n_figure2 = imrotate(n_figure1,3,'crop');% ��ת1��
    n_figure3 = imrotate(n_figure1,-3,'crop');% ��ת-1��
    temp3(:,i) = reshape(n_figure2,256,1);
    temp3(:,i+n) = reshape(n_figure3,256,1);      
end

for i = 1:n
    n_figure0 = train.inputs(:,i);
    n_figure1 = reshape(n_figure0,16,16);
    n_figure2 = imrotate(n_figure1,4,'crop');% ��ת1��
    n_figure3 = imrotate(n_figure1,-4,'crop');% ��ת-1��
    temp4(:,i) = reshape(n_figure2,256,1);
    temp4(:,i+n) = reshape(n_figure3,256,1);      
end

inputs = [train.inputs,double(temp1),double(temp2),double(temp3),double(temp4)];
targets = repmat(train.targets,1,9);
save train_rotate2.mat inputs targets
% save train4.mat inputs targets

%% ���÷Ŵ���С������ѵ����

clear;clc;close all;
load('data/train.mat');
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
save train6.mat inputs targets

%% ���ȴ��ڣ��Ķ���1

clear;clc;close all;
train = load('data/train_rotate2.mat');
[m,n] = size(train.inputs);

figure = train.inputs(:,1:1000);

temp = zeros(m,1000);
for i = 1:m
    for j = 1:1000
        if figure(i,j)>0.4
            temp(i,j) = 1;
        else
            temp(i,j) = 0;
        end
    end
end

inputs = [train.inputs,double(temp)];
targets = [train.targets,train.targets(:,1:1000)];
save train9.mat inputs targets