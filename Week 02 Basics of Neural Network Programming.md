
## Binary Classification 二元分类

对于一张 $m \times m$ 大小的图片，其每个像素通道都可以用RGB的三元组表示，整合为一个输入向量 $x$ ，其维度为 $n_x = m \times m \times 3$

在二元分类中，我们输入这个向量 $x$ ，用神经网络预测对应的输出 $y$ 等于1还是0

*Machine Learning Terminology 机器学习术语*  [Machine Learning Terminology](../MachineLearning2022/Week%2001%20Introduction%20to%20machine%20learning.md#Linear%20Regression%20Model%20线性回归模型)

| 中文术语  | 英文表达  |  英文释义 |
|:---:|:---:|:---:|
|训练集|$training\ set$| Data used to train the model|
|输入变量|$x$|input variable / input feature / feature|
|输出变量|$y$|output variable / target variable|
|估计值|$\hat{y}$|the estimate or the prediction for $y$|
|训练样本总数or训练集实例数量|$m$|number of training examples|
|训练集中的单个实例|$(x,y)$|single training example|
|训练集的第$i$个实例|$(x^{(i)},y^{(i)})$|$i^{\text{th}}$ training example|

可以将 $m$ training examples 的输入 $x^{(1)}, x^{(2)}, \cdots, x^{(m)}$ 合并为一个 $n_x \times m$ 的大的输入矩阵 $X = [x^{(1)}, x^{(2)}, \cdots, x^{(m)}]$
```Python
X.shape = (n_x, m)
```

*在其他表述中，也有将 $X$ 转置表示为一个 $m \times n_x$ 的矩阵的表达*

同理，输出 $y^{(1)}, y^{(2)}, \cdots, y^{(m)}$ 也可以合并为一个 $1 \times m$ 维的矩阵 $Y = [y^{(1)}, y^{(2)}, \cdots, y^{(m)}]$ 
```Python
Y.shape = (1, m)
```

## Logistic Regression 逻辑回归

[Logistic Regression 逻辑回归](../MachineLearning2022/Week%2003%20Classification.md#Logistic%20Regression%20逻辑回归)

Given $x$, want $\hat{y} = \mathcal{P}(y = 1 | x)$

Sigmoid function $g(z) = 1 / (1 + e^{-z})$

$$f_{\vec{w},b}(\vec{x}) = g(\vec{w}\cdot\vec{x} + b) = \frac{1}{1 + e^{-(\vec{w}\cdot\vec{x} + b)}} = \frac{1}{1 + e^{-(w^T x + b)}}$$ 
Is $f_{\vec{w},b}(\vec{x}) \geq 0.5$ (or $z = \vec{w}\cdot\vec{x} + b \geq 0$) ?
- Yes: $\hat{y} = 1$
- No: $\hat{y} = 0$

### Logistic Regression Cost function

[Cost function for Logistic Regression](../MachineLearning2022/Week%2003%20Classification.md#Cost%20function%20for%20Logistic%20Regression)

回顾 cost function for linear regression: 
$$J(\vec{w},b) = \frac{1}{2m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2$$

Loss function $\mathcal{L}(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})$ for the $i^{\text{th}}$ example 
$$\mathcal{L}(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})=
\left
\{
\begin{aligned} 
& -\log(f_{\vec{w},b}(\vec{x}^{(i)})) & \text{if}\ \ y^{(i)} = 1 \\ 
& -\log(1 - f_{\vec{w},b}(\vec{x}^{(i)})) & \text{if}\ \ y^{(i)} = 0
\end{aligned} 
\right.$$

对于 $y^{(i)}$ 不是 $1$ 就是 $0$ 的二元输出，可以重新简化 Loss function 。
the Simplified Cost Function is 
$$\mathcal{L}(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})=
-y^{(i)} * \log(f_{\vec{w},b}(\vec{x}^{(i)})) -(1-y^{(i)}) * \log(1 - f_{\vec{w},b}(\vec{x}^{(i)}))$$

Cost function for Logistic Regression
$$\begin{align*}
J(\vec{w},b) & = \frac{1}{m} \sum_{i=1}^{m}\mathcal{L}(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})  \\
& = -\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} * \log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1-y^{(i)}) * \log(1 - f_{\vec{w},b}(\vec{x}^{(i)}))\right]
\end{align*}$$

## Gradient Descent

[Gradient Descent for Multiple Regression 梯度下降](../MachineLearning2022/Week%2002%20Regression%20with%20multiple%20input%20variables.md#Gradient%20Descent%20for%20Multiple%20Regression%20梯度下降)

find $w$ and $b$ that minimize $J(w, b)$

Simultaneously update 
$$
\begin{align*}
\vec{w} &= \vec{w} - \alpha * \frac{\partial}{\partial \vec{w}} J(\vec{w}, b) = \vec{w} - \alpha * \frac{1}{m} \sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}) * \vec{x}^{(i)} \\
b &= b - \alpha * \frac{\partial}{\partial b} J(\vec{w}, b) = b - \alpha * \frac{1}{m} \sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})
\end{align*}
$$

## Computation Graph 计算图

[Computation Graph](../MachineLearning2022/Week%2005%20Neural%20network%20training.md#Computation%20Graph)

- forward pass 正向传播 进行前向计算，计算神经网络的输出
- backward pass 反向传播 进行反向计算，计算梯度或微分

backward pass $w \rightarrow v \rightarrow u \rightarrow J$
$\frac{\partial{J}}{\partial{w}} = \frac{\partial{J}}{\partial{u}} \times \frac{\partial{u}}{\partial{v}} \times \frac{\partial{v}}{\partial{w}}$

## Vectorization

[A very useful idea **Vectorization** 向量化](../MachineLearning2022/Week%2002%20Regression%20with%20multiple%20input%20variables.md#A%20very%20useful%20idea%20**Vectorization**%20向量化)

Vectorization 使得代码运行速度比 Loop 更快

Whenever possible, avoid explicit **for**-loops.

注：
```Python
a1 = np.array([1, 2, 3, 4, 5])
a2 = np.array([[1, 2, 3, 4, 5]])
```
a1, a2 维度不同：dim(a1) = 1; dim(a2) = 2
```Python
>>> a1.shape
(5,)
>>> a2.shape
(1, 5)
```

一般 Vectorization 时转化为矩阵运算时都会选择后面一种方式，或者通过`reshape`方法变为标准形式 
如：
```Python
a = np.random.randn(5, 1)
b = np.zeros(5).reshape(5, 1)
```
等等，构造 $5 \times 1$ 的列向量，而非 `shape=(5,)` 的 array 数组，前者在进行数学运算时更加符合直觉

一般规定输入 $x$ 为单个输入列向量 $x^{(i)}$ 按行排列形成输入矩阵 $\left[x^{(1)}, x^{(2)}, \cdots, x^{(m)}\right]$，输出的 $\hat{y}$ 为 $1 \times m$ 向量，便于进行后续数据处理。