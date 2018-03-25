function show_data(n)
    % ���ݿ��ӻ�
    % ����n,��ʾѵ�����ݵĵ�n��ͼƬ���ǩ
    close all;

    % ��ȡ����
    %load ('data.mat')
    load('data/train.mat');

    % Your code here
    % ��ȡ�����������
    %n_label0 = train.targets(:,n); % ��ǩ�� 10ά������ n%10ȡģ
    n_figure0 = train.inputs(:,n);

    % ������ת����16x16����ͼƬ��
    n_figure1 = reshape(n_figure0,16,16);
    
    % �Ŵ�ʮ��
    n_figure2 = imresize(n_figure1,[160,160]);
    n_figure2 = imrotate(n_figure2,-90);   % ˳ʱ����ת90��
    n_figure2 = n_figure2(:, end:-1:1);  %���ҷ�ת
    
    % ��ʾͼƬ
    imshow(n_figure2)
    title(['label:' num2str(mod(n,10)-1)]) 
    
end
    

