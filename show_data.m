function show_data(n)
    % 数据可视化
    % 输入n,显示训练数据的第n张图片与标签
    close all;

    % 读取数据
    %load ('data.mat')
    load('data/train.mat');

    % Your code here
    % 读取输入输出向量
    %n_label0 = train.targets(:,n); % 标签： 10维列向量 n%10取模
    n_figure0 = train.inputs(:,n);

    % 将向量转换成16x16矩阵（图片）
    n_figure1 = reshape(n_figure0,16,16);
    
    % 放大十倍
    n_figure2 = imresize(n_figure1,[160,160]);
    n_figure2 = imrotate(n_figure2,-90);   % 顺时针旋转90度
    n_figure2 = n_figure2(:, end:-1:1);  %左右反转
    
    % 显示图片
    imshow(n_figure2)
    title(['label:' num2str(mod(n,10)-1)]) 
    
end
    

