function ret = initial_model(n_hid)
    % ģ�ͳ�ʼ��
    % rand���ȷֲ�
    % normrnd ��̬�ֲ� randn
    n_params = (256+10) * n_hid;
%     as_row_vector = cos(0:(n_params-1));
    % �����ʼ��������һ 
%     eps0 = 0.01;
%     as_row_vector = rand(n_params, 1) * (2*eps0)-eps0;
    % �����ʼ���������� 
%     eps0 = 0.01;
%     as_row_vector = rand(n_params, 1) * eps0;
    % �����ʼ���������� 
%     as_row_vector = rand(n_params, 1) / sqrt(n_params);
    % �����ʼ���������� ��̬�ֲ�
    
%     as_row_vector = randn(n_params, 1) / sqrt(n_params); %0-1֮��
    as_row_vector = (2*randn(n_params, 1)-1) / sqrt(n_params); %-1-1֮��
%     as_row_vector = (randn(n_params, 1)-0.5) / sqrt(n_params); %-0.5-0.5֮��
    
%     as_row_vector = normrnd(0,1,[n_params, 1]) / sqrt(n_params);
%     as_row_vector = normrnd(0,1,[n_params, 1]) * 0.1;
    % �����ʼ����������  
%     as_row_vector = unifrnd (-0.6, 0.6, n_params,1);

%     ƫ��
      

%     ret = theta_to_model(as_row_vector(:));
    ret = theta_to_model(as_row_vector(:) * 1); % ���Կ��ǻ��������ʼ��0.1
    
end