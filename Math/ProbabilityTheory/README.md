--- 概率论 ---



# 贝叶斯公式（Bayes' Theorem）
    P(H|D) = P(D|H) * P_H / P(D_H)


# 最大似然估计（Maximum Likelihood Estimation）


# MAP估计（MAP A Estimation）


# 离散随机变量
    概率函数：P(X = x) = P(X = x | θ)
    伯努利分布：P(X = 1) = θ
    二项分布：P(X = k) = C(n, k) * θ^k * (1 - θ)^(n - k)


# 连续随机变量
    正态分布：P(X = x) = (1 / √2πσ^2) * exp(-(x - μ)^2 / (2σ^2))
        np.random.normal(loc, scale, size)
            loc：均值；
            scale：标准差σ（不是方差）；
            size：输出形状；
            
        np.random.randn(size=None, dtype=None, random_state=None)
            固定生成：标准正态分布 N(μ=0, σ²=1)；
            只能改输出形状，不能改均值、标准差；
            传入数字直接写维度，不是 size 关键字；

    指数分布：P(X = x) = λ * exp(-λx)


# 熵（Entropy）
    熵越高 = 越难预测
    确定事件（P=1）：H = 0（毫无不确定性）
    均匀分布：H 取最大值（最不确定）

    信息量 I(x) = −logP(x)
    熵 H(x) = -sum(P(x) * log(P(x)))
    
