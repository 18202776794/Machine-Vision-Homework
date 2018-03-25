function ret = logistic(input)
    % Your code here
    % 逻辑神经元 logistic neuron
    % ret = input;
    % input是1*1000向量（一个特征/1000个样本）
    % input:隐藏层数目*样本数
    
    [m,n] = size(input);
    ret = zeros(m,n);
    for i=1:m
        for j=1:n
%             ret(i,j) = exp(-(input(i,j)^2));   % Gaussian
%             ret(i,j) = sin(input(i,j));
%             ret(i,j) = log(1.0+exp(input(i,j)));  % softPlus
%             ret(i,j) = atan(input(i,j));  % arctan:反正切: error
%             if input(i,j)<=0        %   ReLU  : error
%                 ret(i,j) = 0;
%             else 
%                 ret(i,j) = 1;
%             end
%             ret(i,j) = 2*1.0/(1.0 + exp(-2*input(i,j))) - 1;  % tanh 函数：error
            ret(i,j) = 1.0/(1.0 + exp(-input(i,j))); % sigmoid 函数
        end
    end
% %     % 第二个隐藏层
%     for i=1:m
%         for j=1:n
%             ret(i,j) = 1.0/(1.0 + exp(-ret(i,j))); % sigmoid 函数
%         end
%     end

end