function ret = d_loss_by_d_model(model, data)
    % �����ݶ�
    % model.input_to_hid is a matrix of size <number of hidden units> by <number of inputs i.e. 256>
    % model.hid_to_class is a matrix of size <number of classes i.e. 10> by <number of hidden units>
    % data.inputs is a matrix of size <number of inputs i.e. 256> by <number of data cases>. Each column describes a different data case. 
    % data.targets is a matrix of size <number of classes i.e. 10> by <number of data cases>. Each column describes a different data case. It contains a one-of-N encoding of the class, i.e. one element in every column is 1 and the others are 0.

    % Your code here.    
    % ���򻯲��� 
    lamda = 0.08;
    
    % �����򴫲���ȡ���������
    % ����
    Xk = data.inputs;
    Sj = model.input_to_hid*Xk;
    Xj = logistic(Sj);
    Si = model.hid_to_class*Xj; 
    exp_Si = exp(Si);
    
    % ��ĸ
    % softmax_sum Ϊ��ĸ 1*500
    % XiΪ���е� ��10*500
    [m,n] = size(data.targets);
    softmax_sum = sum(exp_Si); % sumĬ��������к�
    
    % softmax : 10*500
    softmax = zeros(m,n);
    for i = 1:m
        softmax(i,:) = exp_Si(i,:)./softmax_sum;
    end
    
%     E = - (sum(data.targets.*log(softmax)));
%     classification_loss = mean(E);
    
    % �ٸ�������������ݶ�
    % ret�Ľṹ��model��ͬ������ÿһλ�������ÿ��Ȩֵ���ݶȡ�
    [m2,n2] = size(model.hid_to_class);  % m2--10 output    n2--n ���ز�
    reth = zeros(m2,n2);
    for i2 = 1:m2
        for j2 = 1:n2
            reth(i2,j2) = mean((softmax(i2,:)-data.targets(i2,:)).*Xj(j2,:));
            reth(i2,j2) = reth(i2,j2) + model.hid_to_class(i2,j2)*lamda/n;
        end
    end
%     reth = (softmax-data.targets).*Xj; %  Xj��ת��
%     reti = (model.hid_to_class)'*(softmax-data.targets)*(Xj'*(1-Xj))*Xk';
    [m1,n1] = size(model.input_to_hid); % m1����n   /   n1����256,k
    reti = zeros(m1,n1);
    for j = 1:m1
        for k = 1:n1
             reti(j,k) = mean(sum((softmax - data.targets).*...
                 ((model.hid_to_class(:,j))*((Xj(j,:).*(1-Xj(j,:)))))).*Xk(k,:));
             reti(j,k) = reti(j,k) + model.input_to_hid(j,k)*lamda/n;
        end
    end
      
    
%     ret.hid_to_class = model.hid_to_class * 0;  % ���㵽�����֮��Ȩֵ���ݶ�
%     ret.input_to_hid = model.input_to_hid * 0;  % ����㵽����֮��Ȩֵ���ݶ�
    ret.hid_to_class = reth;
    ret.input_to_hid = reti;
end


