function outimage = myimmove(varargin )
%MYIMMOVE 此处显示有关此函数的摘要
%   x,y 表示偏移的位移大小
%   此处显示详细说明
%参数[80 50］可以修改， 修改后平移距离对应改变
[A, map, x, y] = parse_inputs(varargin{:});


% 如果输入图像不是灰度图，则需要执行转换
if ndims(A)==3,% A矩阵为3维，RGB图像
  A = rgb2gray(A);
elseif ~isempty(map),% MAP变量为非空，索引图像
  A = ind2gray(A,map);
end % 对灰度图像则不需要转换
 
% 读取原始图像的大小并初始化输出图像
[m,n] = size(A);
I = im2double(A);		% 将输入图像转换为双精度类型
outimage = zeros(m,n);

% myimmove(I,x,y ),默认画布不扩大
%函数主体
for i=1:m
    for j=1:n
        if (i-x)>=1&&(j-y)>=1&&(i-x)<=m&&(j-y)<=n
            outimage(i,j)=I(i-x,j-y);
        end
    end
end

if isa(I, 'uint8') % uint8
    outimage = im2uint8(outimage);
elseif isa(I, 'uint16')
    outimage = im2uint16(outimage);
% 其它情况，输出双精度类型的图像
end
end

%-----------------------------------------------------------------------------
function [A, map, x, y] = parse_inputs(varargin)
% 这就是用来分析输入参数个数和有效性的函数parse_inputs
% A       输入图像，RGB图 (3D), 灰度图 (2D), 或者索引图 (X)
% map     索引图调色板 (:,3)
% 首先建立一个空的map变量，以免后面调用isempty(map)时出错
map = [];
 
%   IPTCHECKNARGIN(LOW,HIGH,NUM_INPUTS,FUNC_NAME) 检查输入参数的个数是否
%   符合要求，即NUM_INPUTS中包含的输入变量个数是否在LOW和HIGH所指定的范围
%   内。如果不在范围内，则此函数给出一个格式化的错误信息。
iptchecknargin(2,3,nargin,mfilename);
 
%   IPTCHECKINPUT(A,CLASSES,ATTRIBUTES,FUNC_NAME,VAR_NAME, ARG_POS) 检查给定
%   矩阵A中的元素是否属于给定的类型列表。如果存在元素不属于给定的类型，则给出
%   一个格式化的错误信息。
iptcheckinput(varargin{1},...
              {'uint8','uint16','int16','double'}, ...
              {'real', 'nonsparse'},mfilename,'I, X or RGB',1);
 
switch nargin
 case 2 
  A = varargin{1};
  x = varargin{2}(1);
  y = varargin{2}(2);
 case 3
  A = varargin{1};%               imgrayscaling(X, map, [x1,x2], [y1,y2])
  map = varargin{2};
  x = varargin{2}(1);
  y = varargin{2}(2);
end

% 检测输入参数的有效性
% 检查RGB数组
if (ndims(A)==3) && (size(A,3)~=3)   
    msg = sprintf('%s: 真彩色图像应当使用一个M-N-3维度的数组', ...
                  upper(mfilename));
    eid = sprintf('Images:%s:trueColorRgbImageMustBeMbyNby3',mfilename);
    error(eid,'%s',msg);
end
 
if ~isempty(map) 
% 检查调色板
  if (size(map,2) ~= 3) || ndims(map)>2
    msg1 = sprintf('%s: 输入的调色板应当是一个矩阵', ...
                   upper(mfilename));
    msg2 = '并拥有三列';
    eid = sprintf('Images:%s:inColormapMustBe2Dwith3Cols',mfilename);
    error(eid,'%s %s',msg1,msg2);
    
  elseif (min(map(:))<0) || (max(map(:))>1)
    msg1 = sprintf('%s: 调色板中各个分量的强度 ',upper(mfilename));
    msg2 = '应当在0和1之间';
    eid = sprintf('Images:%s:colormapValsMustBe0to1',mfilename);
    error(eid,'%s %s',msg1,msg2);
  end
end
 
% 将int16类型的矩阵转换成uint16类型
if isa(A,'int16')
  A = int16touint16(A);
end
end

