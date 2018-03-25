function ret = loss(model, data)
    % mean cross-entropy loss.
    % 添加正则化项  L2正则化
    % 正则化系数
    lamda = 0.08;
    
    % Your code here.      
            
    % 先正向传播求取网络输出(softmax)；
    % 分子
    Xk = data.inputs;
    Sj = model.input_to_hid*Xk;
    Xj = logistic(Sj);
    Si = model.hid_to_class*Xj; 
    exp_Si = exp(Si);
    
    % 分母
    % softmax_sum 为分母 1*500
    % Xi为所有的 ：10*500
    [m,n] = size(exp_Si);
    softmax_sum = sum(exp_Si); % sum默认求矩阵列和
    
    % softmax : 10*500
    for j = 1:m
        softmax(j,:) = exp_Si(j,:)./softmax_sum;
    end
    
    % 计算cross-entropy
%     E = - (sum(data.targets.*log(softmax) + (1 - data.targets).*log(1-softmax)));
    E = - (sum(data.targets.*log(softmax)));
    classification_loss = mean(E);
    
    extra = (lamda / (2*n)) * (sum(sum(model.input_to_hid.^2)) + sum(sum(model.hid_to_class.^2)));
    ret = classification_loss + extra;
end