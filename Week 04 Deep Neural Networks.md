Notation:
- $l$ : number of layers
- $n^{[l]}$ : number of units in layer $l$
- $a^{[l]}$ : activations in layer $l$  $a^{[l]} = g^{[l]}(z^{[l]})$ , $a^{[0]} = x$ 
- $w^{[l]}, b^{[l]}$ : parameters of lay. r $l$ for $z^{[l]}$

层数从隐藏层第一层开始算，直至输出层结束，输入层不算在内

第 $l$ 层输入 $a^{[l-1]}$ 并计算输出 $a^{[l]}$ : 
$$
\begin{aligned}
z^{[l]} &= w^{[l]} a^{[l-1]} + b^{[l]} \\
a^{[l]} &= g^{[l]}(z^{[l]})
\end{aligned}
$$

### Getting your matrix dimensions right

- $\text{dim}[w^{[l]}] = n_{in}^{[l]} \times n_{out}^{[l]} = n^{[l-1]} \times n^{[l]}$ 
- $\text{dim}[b^{[l]}] = n_{out}^{[l]} \times 1 = n^{[l]} \times 1$
- $\text{dim}[a^{[l-1]}] = n_{out}^{[l-1]} \times m = n_{in}^{[l]} \times m = n^{[l-1]} \times m$  m 个数据列向量按行堆叠成 m 行矩阵
- $\text{dim}[a^{[l]}] = n_{out}^{[l]} \times m = n^{[l]} \times m$   m 个数据列向量按行堆叠成 m 行矩阵

$$\text{dim}[a^{[l]}] = \text{dim}[g({w^{[l]}}^T a^{[l-1]} + b^{[l]})]$$

## Why deep representations?

拥有众多隐藏层的深度神经网络能够让前面的神经网络层学习较低级别的简单特征，然后让后面更深的神经网络层汇聚前面所检测到的简单信息，以便检测到更复杂的事物。

## Hyperparameters

- parameters: $w^{[l]}$ , $b^{[l]}$ ...
- hyper-parameters: learning rate $\alpha$ , iterations, hidden layers $l$ , hidden units $n^{[l]}$ , choice of activation function, momentum, mini-batch size, regularization ...