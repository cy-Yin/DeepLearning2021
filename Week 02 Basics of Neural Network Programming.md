
## Binary Classification 二元分类

对于一张 $m \times m$ 大小的图片，其每个像素通道都可以用RGB的三元组表示，整合为一个输入向量$x$，其维度为 $n_x = m \times m \times 3$

在二元分类中，我们输入这个向量$x$，用神经网络预测对应的输出$y$等于1还是0

*Machine Learning Terminology 机器学习术语* [Machine Learning Terminology](../MachineLearning2022/Week%2001%20Introduction%20to%20machine%20learning.md#Linear%20Regression%20Model%20线性回归模型)

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

cost function for linear regression: 
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

对于$y^{(i)}$不是$1$就是$0$的二元输出，可以重新简化Loss function。
the Simplified Cost Function is 
$$\mathcal{L}(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})=
-y^{(i)} * \log(f_{\vec{w},b}(\vec{x}^{(i)})) -(1-y^{(i)}) * \log(1 - f_{\vec{w},b}(\vec{x}^{(i)}))$$

Cost function 
$$\begin{align*}
J(\vec{w},b) & = \frac{1}{m} \sum_{i=1}^{m}\mathcal{L}(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})  \\
& = -\frac{1}{m} \sum_{i=1}^{m}[y^{(i)} * \log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1-y^{(i)}) * \log(1 - f_{\vec{w},b}(\vec{x}^{(i)}))]
\end{align*}$$

