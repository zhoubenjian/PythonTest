import torch
import torch.nn as nn
import torch.optim as optim


'''
特征	            含义	              示例值
satisfaction    满意度(0-1)        0.72
projects        参与项目数	      3
hours	        月均加班小时	      45
tenure	        司龄(年)	          2.5
salary	        薪资等级	          0=低, 1=中, 2=高
'''


# 1.定义FNN模型
class Fnn(nn.Module):

    '''
    初始化
    '''
    def __init__(self):
        super().__init__()
        # 输入层 => 隐藏层1
        self.fc1 = nn.Linear(5, 8)
        # 隐藏层1 => 隐藏层2
        self.fc2 = nn.Linear(8, 4)
        # 隐藏层2 => 输出层
        self.fc3 = nn.Linear(4, 1)

        '''
        ReLU（Rectified Linear Unit，修正线性单元）
        公式：ReLU(x) = max(0, x)
        作用：将负数变为0，正数保持不变
        优点：计算简单、缓解梯度消失、产生稀疏激活
        '''
        self.relu = nn.ReLU()       # 通常用于隐藏层

        '''
        Sigmoid 激活函数
        公式：Sigmoid(x) = 1 / (1 + e^(-x))
        输出范围：(0, 1)
        作用：将任意实数映射到0-1之间，适合输出概率
        '''
        self.sigmoid = nn.Sigmoid()     # 通常用于二分类的输出层


    '''
    前向传播
    forward 方法定义前向传播过程
    x：输入数据，形状为 [batch_size, 5]（batch_size个样本，每个样本5个特征）
    '''
    def forward(self, x):
        '''
        第一层：线性变换 => ReLU激活
        self.fc1(x)：将输入通过第一个全连接层，输出形状 [batch_size, 8]
        self.relu(...)：对结果应用ReLU激活函数
        '''
        x = self.relu(self.fc1(x))

        '''
        第二层：线性变换 => ReLU激活
        输入形状 [batch_size, 8]，输出形状 [batch_size, 4]
        '''
        x = self.relu(self.fc2(x))

        '''
        第三层（输出层）：线性变换 → Sigmoid激活
        输入形状 [batch_size, 4]，输出形状 [batch_size, 1]
        Sigmoid将输出压缩到(0,1)区间，表示离职概率
        '''
        x = self.sigmoid(self.fc3(x))

        # 返回预测结果，形状 [batch_size, 1]
        return x


# 2.准备训练数据
'''
X：特征矩阵，形状 (4, 5)
4个样本，每个样本5个特征
dtype=torch.float32：指定数据类型为32位浮点数（神经网络标准精度）
'''
X = torch.tensor([
    [0.72, 3, 45, 2.5, 1],      # 样本1：满意度0.72，项目数3，加班45h，司龄2.5年，薪资中(1)
    [0.38, 5, 68, 1.2, 0],      # 样本2：满意度0.38，项目数5，加班68h，司龄1.2年，薪资低(0)
    [0.91, 2, 38, 4.0, 2],      # 样本3：满意度0.91，项目数2，加班38h，司龄4.0年，薪资高(2)
    [0.45, 4, 72, 0.8, 0]       # 样本4：满意度0.45，项目数4，加班72h，司龄0.8年，薪资低(0)
], dtype=torch.float32)

'''
y：标签向量，形状 (4, 1)
每个样本的真实结果：1表示离职，0表示留任
注意形状是 (4, 1) 而不是 (4,)，这是为了与模型输出形状匹配（都是列向量）
'''
y = torch.tensor([
    [0],        # 样本1：留任
    [1],        # 样本2：离职
    [0],        # 样本3：留任
    [1]         # 样本4：离职
], dtype=torch.float32)


# 3.初始化模型、损失函数、优化器
'''
创建模型实例
这会自动调用__init__方法，初始化所有层的权重和偏置
'''
model = Fnn()

'''
BCELoss：Binary Cross Entropy Loss（二分类交叉熵损失）
适用于二分类问题，标签为0或1
公式：Loss = -[y*log(y_hat) + (1-y)*log(1-y_hat)]
当预测越接近真实标签时，损失越小
'''
criterion = nn.BCELoss()

'''
Adam 优化器（Adaptive Moment Estimation）
一种自适应学习率的优化算法，结合了 Momentum 和 RMSprop 的优点
model.parameters()：获取模型中所有需要训练的参数（权重和偏置）
lr=0.01：学习率（learning rate），控制参数更新的步长
学习率太大：训练不稳定，可能错过最优解
学习率太小：收敛速度慢，可能陷入局部最优
'''
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 4.训练循环
epochs = 1000       # 训练轮数：将整个数据集完整遍历1000次

for epoch in range(epochs):

    # 前向传播
    '''
    将输入数据 X 传入模型，得到预测结果
    outputs 形状：[4, 1]，每个样本的离职概率
    '''
    outputs = model(X)

    '''
    计算损失：比较预测值 outputs 和真实标签 y
    loss 是一个标量张量（0维张量）
    '''
    loss = criterion(outputs, y)

    # 反向传播
    '''
    optimizer.zero_grad()：清除之前累积的梯度
    为什么要清除？PyTorch默认会累加梯度，不清除会导致梯度叠加错误
    '''
    optimizer.zero_grad()

    '''
    loss.backward()：反向传播，计算损失函数对每个参数的梯度
    使用链式法则（自动微分）从输出层向输入层逐层计算梯度
    '''
    loss.backward()

    '''
    optimizer.step()：根据计算出的梯度更新参数
    公式：param = param - lr * grad
    '''
    optimizer.step()

    # 打印训练进度
    # 每100轮打印一次损失值
    if (epoch + 1) % 100 == 0:
        '''
        loss.item()：将只包含一个元素的张量转换为Python标量，便于打印
        f-string：格式化字符串，{表达式:.4f} 表示保留4位小数
        '''
        print(f'Epoch [{epoch + 1} / {epochs}], Loss:{loss.item():.4f}')


# 5.预测新员工
'''
新员工数据：满意度0.65，项目数4，加班55小时，司龄1.8年，薪资中(1)
形状 (1, 5)：1个样本，5个特征
'''
new_employee = torch.tensor([
    [0.65, 4, 55, 1.8, 1]
], dtype=torch.float32)

'''
with torch.no_grad()：上下文管理器，禁用梯度计算
为什么？预测时不需要反向传播，禁用梯度可以节省内存和计算
'''
with torch.no_grad():       # 测试时不需要计算梯度

    # 将新员工数据传入模型，得到离职概率
    prob = model(new_employee)

    '''
    # prob.item()：获取概率值（标量）
    # %.3f：保留3位小数
    '''
    print(f'\n离职概率：{prob.item():.3f}')

    # 根据概率判断结果：>0.5 预测离职，否则预测留任
    print(f"预测结果: {'离职' if prob.item() > 0.5 else '留任'}")





