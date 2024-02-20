## Train/dev/test sets 训练集/开发集/测试集

[Model selection and training / cross validation / test sets](../MachineLearning2022/Week%2006%20Advice%20for%20applying%20machine%20learning.md#Model%20selection%20and%20training%20/%20cross%20validation%20/%20test%20sets)

*开发集(dev sets) 有时也称为 交叉验证集(Cross-validation sets)*

要确保 dev set 和 test set 的数据分布相同

有时没有 test set 也可以

## Bias / Variance 偏差/方差

[Bias and variance 偏差与方差](../MachineLearning2022/Week%2006%20Advice%20for%20applying%20machine%20learning.md#Bias%20and%20variance%20偏差与方差)

high bias 高偏差 -- underfitting
- $J_\text{train}$ is high 训练集拟合得不好
- $J_\text{cv}$ is high 验证集也拟合得不好
high variance 高方差 -- overfitting
- $J_\text{train}$ is low 训练集拟合得很好
- $J_\text{cv}$ is high 验证集拟合得不好
just right:
- $J_\text{train}$ is low
- $J_\text{cv}$ is low

*$J$ 为 Cost function ，用于评估数据集的 error*

### Basic recipe

1. First ask: Does the algorithm have high bias? -> training data performance
	- Bigger network, longer iterations, advanced optimization algorithms ...
2. Then ask: Do you have a variance problem? -> dev set performance
	- More data, regularization ...
3. Done: low bias and low variance

## Regularization 正则化

### L2 regularization

[Regularization 正则化](../MachineLearning2022/Week%2003%20Classification.md#Regularization%20正则化)

[Regularization and bias / variance](../MachineLearning2022/Week%2006%20Advice%20for%20applying%20machine%20learning.md#Regularization%20and%20bias%20/%20variance)

Logistic regression: 
$$\min_{w, b} J(w, b) = \min_{w, b} \left\{\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(y^{(i)}, \hat{y}^{(i)}) + \frac{\lambda}{2m}\Vert w^{[l]}\Vert^2 \right\}$$

其中 $\frac{\lambda}{2m}\Vert w^{[l]}\Vert^2 = \frac{\lambda}{2m}\sum\limits_{j=1}^{n_x}w_j^2 = \frac{\lambda}{2m} w^T w$ 称为 正则化项， $\lambda$ 称为 正则化参数，代码中习惯用 `lambd` 表示。这里取 $w$ 的范数，所以又称为 L2 regularization

在神经网络中， $w$ 成为一个矩阵，Cost function 可以表示为 
$$
J(w^{[1]}, b^{[1]}; w^{[2]}, b^{[2]}; \cdots w^{[L]}, b^{[L]}) = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(y^{(i)}, \hat{y}^{(i)}) + \frac{\lambda}{2m}\sum_{l=1}^{L} \Vert w^{[l]}\Vert_\text{F}^2
$$ 
其中矩阵的范数表示为 $\Vert w^{[l]}\Vert_\text{F}^2 = \sum\limits_{i=1}^{n^{[l-1]}}\sum\limits_{j=1}^{n^{[l]}} \left(w_{ij}^{[l]}\right)^2$  称为 Frobenius norm

$$
\begin{aligned}
\frac{\partial J}{\partial w^{[l]}} &= \text{(from back propagation)} + \frac{\lambda}{m}w^{[l]} \\
w^{[l]} &\coloneqq w^{[l]} - \alpha \frac{\partial J}{\partial w^{[l]}} \\
&=\left(1 - \frac{\alpha \lambda}{m}\right)w^{[l]} - \alpha * \frac{\partial J}{\partial w^{[l]}}' \text{(from old back prop)}
\end{aligned}
$$

$\left(1 - \frac{\alpha \lambda}{m}\right)$ 使得 $w^{[l]}$ 略减小，故又称 L2 regularization 为 Weight decay 权重衰减

### Dropout regularization 随机失活正则化

遍历神经网络的每一层，为网络中的每一个节点设置一个丢弃该节点的概率，即对于神经网络中的每一层，我们将对每一个节点做一次投硬币，使这个节点有 50% 的概率被保留， 50% 的概率被丢弃。全部遍历完成后，决定消除哪些节点，然后清除那些节点和它们进行的计算，得到一个简化很多的神经网络。然后再做反向传播计算。

#### Inverted dropout 反向随机失活

例如：有一个 $l=3$ 的神经网络
`d3` 表示层 3 的失活向量，是一个 bool 矩阵
```Python
d3 = (np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob)
```
设置 `keep_prob` 的值，例如 `keep_prob = 0.8` ，给定节点将被保留的概率值
然后更新 `a3` 为丢弃节点后的输出激活值（通过 `a3` 和 `d3` 逐个元素相乘）
```Python
a3 = np.multiply(a3, d3)
```
最后重新归一化 `a3` ，复原其数学期望值
```Python
a3 = a3 / keep_prob
```

不同的 training examples 实际上对不同的隐藏节点单元进行了置 0 。实际上，如果用一个训练集进行迭代，在不同的训练轮次中置 0 的方式不同，相当于训练了不同的小神经网络。

每一层设置的 `keep_prob` 可以不同，也可以某些层 dropout 而某些层不实施（相当于设置 `keep_prob = 1.0` ），如输入层一般不进行 dropout 。为防止过拟合，一般大的矩阵（对应着有许多参数的层）可以设置小一点的 `keep_prob` 

*测试集上我们不进行 dropout*

#### Understanding dropout

Can't rely on any one feature, so have to spread out weights.

dropout 有个缺点：会使得 Cost function $J$ 不那么明确

### Other regularization methods

1. Data augmentation 增加数据
	- 收录更多数据；将已有图片进行翻转、裁切、变形等处理产生新数据...
2. Early stopping 早终止法
	- 随着 iteration 的增大， train set 的 Cost function 会逐渐减下，但是 dev set 的 Cost function 可能会先减小再增大， Early stopping 在 dev set 的 Cost function 开始增大的那次迭代附近把神经网络的训练过程停下并选取这个最小 dev set error 所对应的值。
	- Early stopping 的缺点是提前终止了神经网络的训练使得神经网络在 train set 中也表现不好

## Normalizing inputs 归一化输入

归一化使得梯度下降寻找 Cost function 的最小值变得容易，可以加速神经网络的训练过程

对于特征 $x$ ，执行 
$$
\begin{aligned}
\mu &= \frac{1}{m}\sum_{i=1}^{m} x^{(i)} \\
\sigma^2 &= \frac{1}{m}\sum_{i=1}^{m} (x^{(i)} - \mu)^2 \\
x^{(i)} &\coloneqq (x^{(i)} - \mu) / \sigma^2
\end{aligned}
$$

对所有特征都这样做，将所有特征全部缩放为 均值为 0，方差为 1

## Vanishing / exploding gradients 梯度的消失和爆炸

当训练层数非常多的神经网络时会遇到 Vanishing / exploding gradients 的问题，即损失函数的导数或斜率非常大或者非常小，使得训练变得困难。

- $w^{[l]} > I$ : leading to exploding gradients
- $w^{[l]} < I$ : leading to vanishing gradients

### Weight initialization for deep network 权值初始化

部分解决 Vanishing / exploding gradients 问题

先看如何初始化单个神经元，再将其应用到深度神经网络

输入 $x = [x_1, x_2, \cdots, x_n]$ 进入单个神经元，输出 $\hat{y} = a = g(z)$ ，其中 $z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n$ （这里忽略 $b$ 项，因为 $b$ 不会对梯度产生很大的影响）。为了不让 $z$ 项太大或者太小，由于加和的关系，项数 $n$ 越大，就希望 $w_i$ 越小。
一个合理的做法是设置 $w_i$ 的 variance 等于 $1 / n$ （如果 activation function 使用 ReLU 函数就设置 variance 等于 $2 / n$ ）
```Python
w_l = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in) 
'''
# if activation function is ReLU
w_l = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
'''
```

## Gradient Checking 梯度检验

帮助确保梯度反向传播的实现是正确的

做梯度检验时我们采用双侧导数 
$$
f'(x) = \lim_{\varepsilon \rightarrow 0}\frac{f(x + \varepsilon) - f(x - \varepsilon)}{2\varepsilon}
$$

1. Take $\{W^{[1]}, b^{[1]}; W^{[2]}, b^{[2]}; \cdots ; W^{[L]}, b^{[L]}\}$ and reshape into a big vector $\theta$
2. Take $\{{\partial J} / {\partial W^{[1]}}, {\partial J} / {b^{[1]}}; {\partial J} / {\partial W^{[2]}}, {\partial J} / {b^{[2]}}; \cdots ; {\partial J} / {\partial W^{[L]}}, {\partial J} / {b^{[L]}}\}$ and reshape into a big vector $d \theta$ ( $\theta$ and $d\theta$ have the same dimension $n$ )
3. Is $d\theta$ the gradient of cost function $J$ ? --> **Gradient Checking** 梯度检验
	For each $i$ from $1$ to $n$ : 
$$
d\theta_\text{approx}[i] = \frac{J(\theta_1, \theta_2, \cdots, \theta_i + \varepsilon, \cdots, \theta_n) - J(\theta_1, \theta_2, \cdots, \theta_i - \varepsilon, \cdots, \theta_n)}{2\varepsilon}
$$
4. Check if these two vectors $d\theta$ and $d\theta_\text{approx}$ are approximately equal to each other by  Euclidean distance 检验距离（范数）
$$
\frac{\Vert d\theta - d\theta_\text{approx}\Vert_2}{\Vert d\theta_\text{approx}\Vert_2 + \Vert d\theta \Vert_2} \leq 10^{-7}
$$
### Gradient checking implementation  notes

- 不要在训练中使用，只在 Debug 时使用
- 如果一个算法没有通过 Gradient Checking ，需要检查每一个组成成分尝试找出漏洞 -- 具体地，检查每一个 $i$ 分量，从而找出是哪一个 $w$ 或 $b$ 或它们的分量
- 如果使用了正则项，不要忘记计算 $d\theta$ 和 $d\theta_\text{approx}$ 时带上正则项
- Gradient Checking 和 dropout 不能同时使用
- 有时初始化 $w$ 和 $b$ 时 Gradient Checking 是好的，但是梯度下降一段时间后 $w$ 和 $b$ 会产生变化，所以在训练一段时间后可以再次进行 Gradient Checking