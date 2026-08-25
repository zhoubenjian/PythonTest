'''
最大似然估计(Maximum Likelihood Estimation, MLE)

最大后验估计(Maximum A Posteriori Estimation, MAP)


两者区别：
    最大似然估计(MLE)是基于观测数据的，而最大后验估计(MAP)则同时是基于观测数据和先验知识的；

    MAP = MLE + 先验 P(θ)，当先验是均匀分布时，MAP = MLE，当先验是非均匀分布时，MAP = MLE + 先验 P(θ)；

    概率 P(数据|参数)：参数固定，问「在这个参数下，观察到这些数据的概率多大？」
    似然 L(参数|数据)：数据固定，问「哪个参数值最可能产生这些数据？」
    似然函数的值本身不归一（不要求和为 1），它的意义在于 比较不同参数值的相对合理性。


MLE 和 MAP 是统计推断的两种核心方法，训练神经网络的本质 ≈ 最大似然估计
'''
import numpy as np


# 固定随机种子，确保结果可重复
np.random.seed(42)
# 初始化 μ = 5, σ = 2 的数据集
data = np.random.normal(5, 2, 100)


'''
# MLE: μ = 5, σ = 2
'''
μ_mle = np.mean(data)
print(f"MLE: μ = {μ_mle:.0f}(真实值=5.0)")     # MLE: μ = 5(真实值=5.0)
σ_mle = np.std(data, ddof=0)
print(f"MLE: σ = {σ_mle:.0f}(真实值=2.0)")     # MLE: σ = 2(真实值=2.0)


'''
MAP: 先验 μ ~ N(0, 1)，向先验方向收缩
'''
prior_mu, prior_var = 0.0, 1.0
n = len(data)

# MAP均值 = (先验均值 / 先验方差 + n * MLE均值 / MLE方差) / (1 / 先验方差 + n / MLE方差)
mu_map = (prior_mu / prior_var + n * μ_mle / σ_mle ** 2) / (1 / prior_var + n / σ_mle ** 2)
print(f"MAP: μ = {mu_map:.3f}(在MLE和先验0之间折中)")       # MAP: μ = 4.641(在MLE和先验0之间折中)

# MAP方差 = 1 / (1 / 先验方差 + n / MLE方差)
var_map = 1 / (1 / prior_var + n / σ_mle ** 2)
print(f"MAP: σ = {np.sqrt(var_map):.3f}(真实值=2.0)")     # MAP: σ = 0.178(真实值=2.0)


