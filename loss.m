function ret = loss(model, data)
    % mean cross-entropy loss.
    % ���������  L2����
    % ����ϵ��
    lamda = 0.08;
    
    % Your code here.      
            
    % �����򴫲���ȡ�������(softmax)��
    % ����
    Xk = data.inputs;
    Sj = model.input_to_hid*Xk;
    Xj = logistic(Sj);
    Si = model.hid_to_class*Xj; 
    exp_Si = exp(Si);
    
    % ��ĸ
    % softmax_sum Ϊ��ĸ 1*500
    % XiΪ���е� ��10*500
    [m,n] = size(exp_Si);
    softmax_sum = sum(exp_Si); % sumĬ��������к�
    
    % softmax : 10*500
    for j = 1:m
        softmax(j,:) = exp_Si(j,:)./softmax_sum;
    end
    
    % ����cross-entropy
%     E = - (sum(data.targets.*log(softmax) + (1 - data.targets).*log(1-softmax)));
    E = - (sum(data.targets.*log(softmax)));
    classification_loss = mean(E);
    
    extra = (lamda / (2*n)) * (sum(sum(model.input_to_hid.^2)) + sum(sum(model.hid_to_class.^2)));
    ret = classification_loss + extra;
end