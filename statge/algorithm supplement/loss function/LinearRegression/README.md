# 线性回归损失函数



# 均方误差损失函数(Mean Squared Error)L2 Loss
    Loss(y_pred, y_true) = 1/2 * (y_pred - y_true) ** 2

    MSE 假设误差服从高斯分布，对大误差惩罚重，对小误差惩罚轻
    MSE 对离群点敏感


# 绝对值损失函数(Mean Absolute Error)L1 Loss
    Loss(y_pred, y_true) = |y_pred - y_true|

    MAE 假设误差服从拉普拉斯分布，对所有误差一视同仁
    MAE 鲁棒但梯度不连续



# Huber Loss
    Loss(y_pred, y_true, δ) = 1/2 * (y_pred - y_true) ** 2          if |y_pred - y_true| <= δ
    Loss(y_pred, y_true, δ) = δ * (|y_pred - y_true| - δ * 0.5)     if |y_pred - y_true| > δ
    其中δ 是一个超参数（通常取 1.0），控制"小误差"和"大误差"的分界线

    Huber Loss 对离群点不敏感，同时保持了对小误差的敏感性

    δ通常取 1.0。如果数据误差普遍较大，可以调大；如果误差普遍很小，调小。也可以用分位数（如 90% 分位）动态设定



# L1 正则（Lasso）vs L2 正则（Ridge）

    线性回归基础损失：MSE = 1/n * Σ(y_pred - y_true)²
    正则化就是在 MSE 后面"加一项惩罚"，目的：抑制过拟合 + 控制模型复杂度

    统一形式：Loss = MSE + α * penalty(β)
        α 越大，惩罚越重，模型越简单；α = 0 退化为普通线性回归


## L2 正则 —— Ridge（岭回归）

    惩罚项：penalty = Σ βᵢ²          （系数的平方和）
    完整损失：Loss = MSE + α * Σ βᵢ²

    特点：
        1. 把所有系数往 0 方向"缩小"，但不会真的变成 0（始终保留所有特征）
        2. 连续可导，优化稳定，有闭式解（β = (XᵀX + αI)⁻¹ Xᵀy）
        3. 对特征共线性（多重共线性）鲁棒——这是 Ridge 最核心的价值
        4. α 越大，系数越趋近 0（但不是 0），偏差增加、方差降低

    几何直观：L2 约束是一个超球。等高线与球相切时，系数一般不会落在轴上，所以不为 0


## L1 正则 —— Lasso（套索回归）

    惩罚项：penalty = Σ |βᵢ|        （系数的绝对值之和）
    完整损失：Loss = MSE + α * Σ |βᵢ|

    特点：
        1. 能把部分系数直接压到 0 → 天然具有"特征选择"能力，得到稀疏模型
        2. 绝对值在 0 处不可导，没有闭式解，需要坐标下降 / 近端梯度等迭代算法
        3. 对离群点相对鲁棒（对大系数的惩罚是线性的，不像平方那样爆炸性增长）
        4. 多重共线性时：Lasso 倾向于从相关特征里"挑一个"保留，其余压 0

    几何直观：L1 约束是一个超菱形（有尖角）。等高线与菱形相切时更容易碰到"角"（在某个轴上），所以系数变 0


## 对比一览

    | 维度             | L2 Ridge            | L1 Lasso              |
    |------------------|---------------------|-----------------------|
    | 惩罚项           | Σ βᵢ²                | Σ |βᵢ|                 |
    | 系数是否归零      | 不会（整体缩小）     | 会（部分归零）          |
    | 特征选择能力      | 无                  | 有（自动稀疏化）        |
    | 闭式解            | 有                  | 无（迭代求解）          |
    | 对共线性的处理    | 分摊到所有相关特征    | 保留一个、其余归零       |
    | 数值稳定性        | 高                  | 中（解路径可能不连续）   |
    | 训练速度          | 快                  | 较慢（迭代）            |


## 使用场景

    用 Ridge（L2）当：
        - 特征之间存在多重共线性（这是 L2 最擅长的场景）
        - 你想保留所有特征、只是想让系数更保守
        - 想要稳定、有解析解、训练快
        - 样本量小、特征数不太多时防过拟合

    用 Lasso（L1）当：
        - 你想做特征选择（高维稀疏数据，尤其 p >> n）
        - 希望模型可解释（只保留少数关键特征）
        - 需要稀疏解以节省存储/推理开销
        - 怀疑大部分特征是噪声

    两个都想用 → Elastic Net（L1 + L2 混合）：
        Loss = MSE + α * ( ρ*Σ|βᵢ| + (1-ρ)*Σβᵢ² )
        ρ 控制 L1/L2 比例。Elastic Net 解决了 Lasso 在强相关特征上"随机挑一个"的问题，
        能把相关特征一起保留或一起归零。高维数据优先考虑它。


## sklearn 用法

    from sklearn.linear_model import Ridge, Lasso, ElasticNet

    Ridge(alpha=1.0, max_iter=1000, random_state=42)
    Lasso(alpha=1.0, max_iter=1000, random_state=42)
    ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, random_state=42)

    调参顺序：先用大网格找量级（1e-3, 1e-2, ..., 1e2），再在附近细化。
    注意：Ridge/Lasso 都要求特征标准化后再训练，否则惩罚对大尺度特征不公平。


## 一句话记忆

    L2 是"把所有人的工资都扣一点"——大家都在，但都变少了；
    L1 是"绩效不合格的直接开除"——留下的人就是真正重要的特征。


# Tips
    TensorFlow/Keras 和 Scikit-learn 选择了 y_true, y_pred （真实值在前）
    PyTorch 则选择了 y_pred, y_true （预测值在前）
    不过，这个规律在 PyTorch 内部也有例外。例如，当你使用 torch.nn.functional.binary_cross_entropy 时，它的参数顺序是 (input, target)，也就是 (y_pred, y_true) 。但在更早的 nn.BCELoss 中，又是 (input, target)。所以，PyTorch 的底层 functional 接口倾向于 y_pred, y_true，但 nn.Module 的封装类通常用 input, target 来命名，这是一种更通用的表述。
