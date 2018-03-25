function ret = logistic(input)
    % Your code here
    % �߼���Ԫ logistic neuron
    % ret = input;
    % input��1*1000������һ������/1000��������
    % input:���ز���Ŀ*������
    
    [m,n] = size(input);
    ret = zeros(m,n);
    for i=1:m
        for j=1:n
%             ret(i,j) = exp(-(input(i,j)^2));   % Gaussian
%             ret(i,j) = sin(input(i,j));
%             ret(i,j) = log(1.0+exp(input(i,j)));  % softPlus
%             ret(i,j) = atan(input(i,j));  % arctan:������: error
%             if input(i,j)<=0        %   ReLU  : error
%                 ret(i,j) = 0;
%             else 
%                 ret(i,j) = 1;
%             end
%             ret(i,j) = 2*1.0/(1.0 + exp(-2*input(i,j))) - 1;  % tanh ������error
            ret(i,j) = 1.0/(1.0 + exp(-input(i,j))); % sigmoid ����
        end
    end
% %     % �ڶ������ز�
%     for i=1:m
%         for j=1:n
%             ret(i,j) = 1.0/(1.0 + exp(-ret(i,j))); % sigmoid ����
%         end
%     end

end