优化器
├── 基础理论
│   ├── 梯度下降的数学原理
│   ├── 凸优化 vs 非凸优化
│   ├── 局部最优、鞍点、平坦/尖锐极小值
│   └── 学习率的作用与困境
├── 一阶方法（主流）
│   ├── BGD / SGD / Mini-batch SGD
│   ├── Momentum
│   ├── Nesterov Momentum
│   ├── AdaGrad
│   ├── RMSProp
│   ├── Adam
│   ├── AdaDelta
│   ├── AdamW
│   └── 变体：Nadam / RAdam / LAMB / Lookahead 等
├── 二阶方法（了解即可）
│   ├── 牛顿法
│   ├── 拟牛顿法（BFGS / L-BFGS）
│   └── 自然梯度 / K-FAC
├── 学习率调度
│   ├── Step / MultiStep
│   ├── Exponential / Cosine
│   ├── Warmup
│   ├── OneCycle
│   └── ReduceLROnPlateau
└── 实践技巧
    ├── 梯度裁剪
    ├── 权重衰减 vs L2 正则
    ├── Batch Size 与学习率的缩放关系
    └── 优化器选择经验