## Mini-batch gradient descent 小批量梯度下降

[Algorithm refinement Mini-batch and soft update 小批量和软更新](../MachineLearning2022/Week%2010%20Reinforcement%20learning.md#Algorithm%20refinement%20Mini-batch%20and%20soft%20update%20小批量和软更新)

加速神经网络训练

将大数据集拆分成小的 mini-batch，如对于 $m = 5,000,000$ ，拆分成 5,000 个 包含 $m' = 1,000$ 个样例的小数据集，第 $t$ 个 mini-batch 记为 $\{x^{\{t\}}, y^{\{t\}}\}$ ，$x^{\{t\}}$ 的维度为 $(n_x, m')$ ，$y^{\{t\}}$ 的维度为 $(1, m')$

mini-batch 的思想就是
For $t$ from $1$ to $5,000$
{
$$
\begin{aligned}
w &= w - \alpha * \frac{\partial}{\partial w} \frac{1}{2m'} \sum_{i=1}^{m'} \left( f_{w,b}({x^{\{t\}}}^{(i)}) - {y^{\{t\}}}^{(i)} \right)^2 \\ 
b &= b - \alpha * \frac{\partial}{\partial b} \frac{1}{2m'} \sum_{i=1}^{m'} \left( f_{w,b}({x^{\{t\}}}^{(i)}) - {y^{\{t\}}}^{(i)} \right)^2
\end{aligned}
$$
}
*在 5000 轮训练下来，完整经过了一遍数据集，称为一次 **epoch***

通过 mini-batch ，原本一次 epoch 只能进行一次 Gradient descent，而现在可以进行 5000 次，大大加速了训练效率

### Understand Mini-batch gradient descent

Choosing your mini-batch size:
- If mini-batch size is $m$ : Batch gradient descent $\{x^{\{1\}}, y^{\{1\}}\} = \{x, y\}$
	- Too long per iteration for a big dataset
- If mini-batch size is $1$ : Stochastic gradient descent 随机梯度下降 Every example is its own mini-batch
	- noise; lose speedup from vectorization
- In practice : Somewhere in between $1$ and $m$ (mini-batch size not too big/ small)
	- Vectorization
	- Make progress without needing to wait till you process the entire training set

具体如何选择：
- small training set ( $m \leq 2000$ ) : no mini-batch, just batch gradient descent
- typical mini-batch size: $64, 128, 256, 512$
- Make sure mini-batch fit $\{x^{\{t\}}, y^{\{t\}}\}$ fit in CPU / GPU memory

