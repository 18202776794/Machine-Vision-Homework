function ret = initial_model(n_hid)
    % 模型初始化
    % rand均匀分布
    % normrnd 正态分布 randn
    n_params = (256+10) * n_hid;
%     as_row_vector = cos(0:(n_params-1));
    % 随机初始化：方法一 
%     eps0 = 0.01;
%     as_row_vector = rand(n_params, 1) * (2*eps0)-eps0;
    % 随机初始化：方法二 
%     eps0 = 0.01;
%     as_row_vector = rand(n_params, 1) * eps0;
    % 随机初始化：方法三 
%     as_row_vector = rand(n_params, 1) / sqrt(n_params);
    % 随机初始化：方法四 正态分布
    
%     as_row_vector = randn(n_params, 1) / sqrt(n_params); %0-1之间
    as_row_vector = (2*randn(n_params, 1)-1) / sqrt(n_params); %-1-1之间
%     as_row_vector = (randn(n_params, 1)-0.5) / sqrt(n_params); %-0.5-0.5之间
    
%     as_row_vector = normrnd(0,1,[n_params, 1]) / sqrt(n_params);
%     as_row_vector = normrnd(0,1,[n_params, 1]) * 0.1;
    % 随机初始化：方法五  
%     as_row_vector = unifrnd (-0.6, 0.6, n_params,1);

%     偏置
      

%     ret = theta_to_model(as_row_vector(:));
    ret = theta_to_model(as_row_vector(:) * 1); % 可以考虑换成随机初始化0.1
    
end