# PythonTest
Learning artificial intelligence...



pip install 包名 -i https://pypi.tuna.tsinghua.edu.cn/simple
清华源：https://pypi.tuna.tsinghua.edu.cn/simple
腾讯：https://mirrors.cloud.tencent.com/pypi/simple
阿里：https://mirrors.aliyun.com/pypi/simple/




# 1.数学：
    *线性代数*：
        -特征值和特征向量（只适用于方阵）：Av = λv
            特征向量（v）： 一个非零向量，被矩阵 A 作用后，仅仅被拉伸或压缩，方向不变；
            特征值（λ）： 拉伸/压缩的倍数；

        -奇异值分解（适用于任意矩阵）：A = U @ Σ @ V(t)
            U：左奇异向量（m×m），正交矩阵；
            Σ：奇异值在对角线上（m×n），对角矩阵，奇异值从大到小排列；
            V(t)：右奇异向量的转置（n×n），正交矩阵；



# 神经网络：
    每条连接线（箭头）配一个w，用来调节信号强弱。
    每个细胞体（圆圈）配一个b，用来调节自身激活门槛。