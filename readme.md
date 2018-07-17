# 项目总结：神经网络手写数字识别

标签（空格分隔）： 笔试面试

---


## 小结
+ 该项目主要利用matlab开发的基于三层神经网络的手写数字识别
+ 样本数 1000
+ 随机初始化，（-1，1）高斯分布，使用1/sqrt(n)校准方差
+ 样本数据增强：旋转/缩放/平移/二值化？？
+ 训练：计算梯度（经过反向传播公式推导）

## MATLAB代码框架
1. show_data(): 显示数据
2. initial_model：随机初始化
3. theta_to_model/model_to_theta：数据向量矩阵化/矩阵向量化——Flatten
4. loss：前向传播求softmax-交叉熵损失
5. d_loss_by_d_model：计算梯度；前向传播计算对数损失（softmax）;反向传播
6. logistic: sigmoid激活函数
7. test_gradient：梯度检测
8. evaluation 

# 一、	输出结果
 ![image_1ca7689nb9r41beb103t1mfj10hg9.png-8kB][1]
 
 ![image_1ca769e051mpjemgode177h1dmqm.png-13.8kB][2]
 
# 二、	算法说明
##1.	图片操作
+ show_data函数主要负责完成图像的显示。程序如下方所示：
 
![图1][3]
 
+ 函数的参数n代表数据的第n列，即第n张训练集图像。

1. 首先，通过reshape函数将将提取的第n列数据n_figure0转化为16*16的矩阵n_figure1，也就是所对应的图像。

2. 然后我们通过imresize函数将得到的图像矩阵变换为160*160的图像。

3. 此时如果直接显示就会出现图2中左图的情况，因此我通过imrotate函数将图像顺时针旋转90度，然后左右反转（也就是列反向排序）。

+ one hot标签： 另外，我们需要显示图像标签，也即图像显示的数字。从下图我们可以看出标签10个一循环。且经过实验可知顺序是按照0-9。故将所取的列号对10取模，然后减1即可得当前图像的标签。
 
##2.	激活函数
本文中神经网络为三层全连接神经网络，结构图如下所示：

 ![image_1ca76hetm1kti9h6ki49r12gf1t.png-38.4kB][4]

激活函数主要再logistic文件中完成，实现隐藏层的传递运算。其中主要程序如下：

``` matlab
[m,n] = size(input);
ret = zeros(m,n);

for i=1:m
    for j=1:n
        ret(i,j) = 1.0/(1.0 + exp(-input(i,j))); % sigmoid 函数
    end
end
```

本算法中的神经网络的隐藏层采用了逻辑元（logistic neuron ），我们选择sigmoid激活函数。公式如下所示：
 ![image_1ca76m7n7dd31gcf16l11b3elee2a.png-6.2kB][5]

首先，通过断点调试，我们可以发现。Sigmoid函数的输入input大小为n_hid（隐藏层数）*1000（样本数），即Sj，表示的是经过权值处理后的隐藏层输入。

然后，我们通过双层for循环来完成对每一个样本的各个隐藏层节点进行sigmoid函数的运算。Ret即为计算后的隐藏层输出。

## 3.	误差计算
误差计算主要通过loss函数实现，这个函数是用来计算训练数据的误差。多类分问题的误差 E常选为 cross-entropy。（为了解决sigmoid两端误差更新太慢问题，偏导太小）

Loss函数输入当前模型model，和训练数据data，通过前向传播计算网络输出结果（softmax结果），

计算每个样本的交叉熵cross-entropy E。并输出所有样本的评价误差。

Softmax和cross-entropyE的计算公式如下：

+ Softmax分类：![image_1ca76pj133r31h3614jsldu1svh2n.png-6.7kB][6]
 
+ 交叉熵损失: ![image_1ca76q3f3ol51oem1rs411q71j3634.png-4.5kB][7]； t是标签， y是估计值 

+ 总样本损失： ![image_1ca76qchc1rjg150t6jt1e3l13fs3h.png-7.2kB][8]

由于有多个训练样本，最后所求的误差loss实际上是多个样本的均值。

> 拓展： 关于softmax & 交叉熵损失
[Softmax 函数的特点和作用是什么？](https://www.zhihu.com/question/23765351/answer/139826397)
[交叉熵+softmax](https://www.cnblogs.com/golearning/p/6814427.html)

算法实现：

+ 首先，我们按顺序依次求出输入层、隐藏层以及输出层的输入和输出。代码如下所示。Xk表示输入层输入，Sj表示乘上权值以后隐藏层的输入，Xj表示隐藏层的输出，Si表示乘上权值后输出层的输入。

```matlab
Xk = data.inputs;
Sj = model.input_to_hid*Xk;
Xj = logistic(Sj);
Si = model.hid_to_class*Xj; 
exp_Si = exp(Si);
[m,n] = size(exp_Si);
softmax_sum = sum(exp_Si); % sum默认求矩阵列和
```
+ 由于有1000个样本，所以上述操作均是对矩阵的操作，也就是对1000个样本进行相同的操作。其中exp_Si和softmax_sum分别是softmax公式中的分子和分母。因此softmax如下所示：
 ![image_1ca7739dd9dvvo1a9u1jsk1uqt3u.png-2.5kB][9]

+ 而相应的cross-entropy以及最终误差由公式可求，代码如下：
 ![image_1ca773pss3li5giqu71fsj1ppu4b.png-2.7kB][10]

+ 由于是各样本中各对应元素进行计算，所以需要特别注意上述算法需要用点乘或是点除运算。最终函数的返回值为classification_loss，即各样本误差的平均值。

## 4.	梯度计算
当我们计算得到当前网络的误差时，就就可以通过求导的链式法则将误差反向传播到每个网络权值处，用于修改权值。

$W_{kj}$表示输入层到隐层连接权重，$W_{ji}$表示隐层到输出层的连接权重，$S_j$表示隐层的输入。$X_j$表示隐层的输出，$S_i$表示输出层的输入，$X_i$是输出层的输出（每类误差），

则误差关于每个权值的导数计算公式如下：(**推导见另一篇md**)
![image_1ca77rjm3989sud1mi53ff14lu55.png-38.3kB][11]
 
+ 程序实现:
1. 首先，通过前向传播计算网络输出结果（softmax结果），计算每个样本的交叉熵cross-entropy E，如loss中程序所示。

2. 其次，根据所得误差计算相对于权值的误差。

![图5][12]

由于公式较为复杂，上述程序主要是通过两层循环对两处权值分别进行遍历，然后按照公式计算。Reth表示误差相对于hid_to_input权值的导数，而reti表示误差相对于input_to_hid的导数。需要特别注意的是公式中的下标，公式是相对于每一个元素，因此程序中应该使用点乘或者点除。

## 5.	调节网络参数
+ 需要调节的网络参数主要包括以下四个参数：
 + n_hid:  是隐层神经元数量，隐层神经元数量越多，网络的拟合能力越强，但计算速度也将下降。
 
 + n_iters:  是神经网络训练的迭代次数，迭代次数越多，对训练集的使用越充分。
 
 + learning_rate: 学习率。学习率可以控制权值更新的速度。
 
 + mini_batch_size: 是批尺寸: 表示每次迭代计算的样本数。

###  学习率迭代次数 
+ 由于学习率不宜过大，应尽可能小；但是学习率过小，须更多次迭代导致计算时间太慢，因此经过多次试探，初步选择学习率为0.3而迭代次数为900。而由图可知，迭代900次，还未完全收敛。在确定最终其他参数后，我将迭代次数设为3000次，以确定最终收敛到最小误差的值。

![图6 迭代900次][13]

### 隐藏层节点数

+ 而关于隐藏层节点数，初步选择时主要参照经验公式，但是节点数的选择并不与下列经验公式完全一致。经过反复试探，最终将隐藏层节点数确定为40。
 
### 批处理数mini_batch_size
+ 而关于批处理数，虽然建议采用full_batch，但我发现批尺寸选择在125或者200左右比选择full_batch时效果略好，因此最终批尺寸选为200。应该特别注意批尺寸应为样本数的因数。？？


## 6.	改进措施
### A.	随机初始化权值
> 参考博客：
[神经网络权重初始化问题](https://blog.csdn.net/marsggbo/article/details/77771497)
[为什么神经网络在考虑梯度下降的时候，网络参数的初始值不能设定为全0，而是要采用随机初始化思想？](https://www.zhihu.com/question/36068411?sort=created)

    小结：
    1. 若初始权值相同为0，则由反向传播求导公式可知，训练出来的权重也相同
    2. 若初始值为小的随机数,随机初始化神经元的输出的分布有一个随输入量增加而变化的方差
    3. 我们可以通过将其权重向量按其输入的平方根(即输入的数量)进行缩放，从而将每个神经元的输出的方差标准化到1
    4. 这保证了网络中所有的神经元最初的输出分布大致相同，并在经验上提高了收敛速度。


原先的权值初始化主要通过cos函数选取并乘系数0.1弱化，因此之前的初始权值是固定的。而一种常用的方法是对各个权值使用均值为0方差为1的正态分布的随机值。***为什么？？？***

方差归一化：如果神经元刚开始的时候是随机且不相等的，那么它们将计算出不同的更新，并将自身变成整个网络的不同部分。随着输入数据量的增长，随机初始化的神经元的输出数据的分布中的方差也在增大。我们可以除以输入数据量的平方根来调整其数值范围，这样神经元输出的方差就归一化到1了。

因此，我们可以使用1/sqrt(n)校准方差。初始化的代码如下所示：
```MATLAB
as_row_vector = (2*randn(n_params, 1)-1) / sqrt(n_params); % -1-1之间
```
注意还需通过线性变换将随机值范围变为-1~1。

### B.	L2正则化(另一篇博客也有写)
为了减小过拟合，我采用了L2正则化的方法。L2正则化就是在代价函数后面再加上一个正则化项：
 ![image_1ca78mr6a1mct1i2cmnkvjfbg82.png-6.5kB][14]

则相应梯度计算时也应该在原有基础上增加一项：
 ![image_1ca78p0ch1t8vigjicq1k4cfmk8f.png-7.6kB][15]

而相应的程序则加上正则项：正则化系数lamda经过多次试验，最终确定为0.08.

L2正则化有让权值变小的效果，更小的权值w，从某种意义上说，表示网络的复杂度更低，对数据的拟合刚刚好。

### C.	增加样本数量
1. 原始程序中只使用了500组样本，这是显然不够的。首先，我将另外500组训练样本也加入。

2. 然后尝试了平移、旋转、缩放以及二值化增强等方法来增加训练集。结果发现，平移和缩放会导致结果明显下降，因此这两种方法不予采用。

3. 最终，我将原始1000组数据分别顺、逆时针旋转1，2，3，4度，这样共得到9000组数据，

4. 数据预处理：增强亮度； 另外，以0.4为阈值将原始数据进行二值化，得到另外1000组数据，总共10000组数据，再来进行处理。

5. 但是在实际测试中，经过对比，我剔除	了旋转2、3、4度的数据，以训练出较好的模型。添加样本的程序见文件data_add.m


  [1]: http://static.zybuluo.com/QQJayden/4mug58zpm95xit06xonfdfew/image_1ca7689nb9r41beb103t1mfj10hg9.png
  [2]: http://static.zybuluo.com/QQJayden/p9on6fokhacjziojnln0hxez/image_1ca769e051mpjemgode177h1dmqm.png
  [3]: http://static.zybuluo.com/QQJayden/8raq3zgrtv17dob5shq5tg7u/image_1ca76bhva15ed1p8m1oonoqlsbd13.png
  [4]: http://static.zybuluo.com/QQJayden/eeji6kv6x6y1s8273qbiyu1c/image_1ca76hetm1kti9h6ki49r12gf1t.png
  [5]: http://static.zybuluo.com/QQJayden/rhxn7jdllr4v6ckj21zbpkou/image_1ca76m7n7dd31gcf16l11b3elee2a.png
  [6]: http://static.zybuluo.com/QQJayden/85can6c66gx0zmwryvyv7lhl/image_1ca76pj133r31h3614jsldu1svh2n.png
  [7]: http://static.zybuluo.com/QQJayden/564ijlxxw4dp3d09yqbi8q3g/image_1ca76q3f3ol51oem1rs411q71j3634.png
  [8]: http://static.zybuluo.com/QQJayden/0txxfx0flaszwq6dsqosumkq/image_1ca76qchc1rjg150t6jt1e3l13fs3h.png
  [9]: http://static.zybuluo.com/QQJayden/v3if0fc9wcresnzjhhuz1rpq/image_1ca7739dd9dvvo1a9u1jsk1uqt3u.png
  [10]: http://static.zybuluo.com/QQJayden/pmtf2z584eqsrobi90sw92c7/image_1ca773pss3li5giqu71fsj1ppu4b.png
  [11]: http://static.zybuluo.com/QQJayden/b9b0rnmaqg0g3deu3q5apze6/image_1ca77rjm3989sud1mi53ff14lu55.png
  [12]: http://static.zybuluo.com/QQJayden/kregb7kgrkui4hgbsraqmotu/image_1ca77un931704ofkg1hn3f8qr5i.png
  [13]: http://static.zybuluo.com/QQJayden/gm5oyrxq3nrvajy5miq5xtpp/image_1ca784ugv23nopu1ob61m90141g7i.png
  [14]: http://static.zybuluo.com/QQJayden/mdz3qd0kxl6k44u1qbrvlgei/image_1ca78mr6a1mct1i2cmnkvjfbg82.png
  [15]: http://static.zybuluo.com/QQJayden/ylxdvxmdtbmquf08v2bttugf/image_1ca78p0ch1t8vigjicq1k4cfmk8f.png