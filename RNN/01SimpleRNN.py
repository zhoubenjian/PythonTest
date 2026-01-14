'''
PyTorch实现简单循环神经网络（RNN）
'''
import torch
import torch.nn as nn
import torch.optim as optim


# 1.定义超参数（序列相关+模型相关）
sequence_length = 10    # 序列长度（每个序列有10个时间步）
input_size = 5          # 序列长度（每个序列有10个时间步）
hidden_size = 16        # RNN 隐藏状态的特征维度
num_layers = 1          # RNN 层的数量（基础 RNN 可堆叠多层）
batch_size = 3          # 批次大小（一次处理3个序列）
output_size = 1         # 最终输出维度（序列到标签任务，每个序列输出一个标量）


# 2.定义简单RNN模型（基础RNN + 全连接输出）
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义基础RNN（Pytorch内置nn.RNN）
        self.rnn = nn.RNN(
            input_size = input_size,    # 每个时间步输入特征维度
            hidden_size = hidden_size,  # 隐藏状态特征维度
            num_layers = num_layers,    # RNN 层数
            batch_first = True          # 设为 True 时，输入形状为 [batch_size, sequence_length, input_size], 默认为 False，输入形状为 [sequence_length, batch_size, input_size]
        )

        # 定义全连接层，将RNN最终的隐藏状态映射到任务输出唯独
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 步骤1：初始化隐藏状态 h0（初始时刻的隐藏状态，全 0 张量）
        # 形状：[num_layers * num_directions, batch_size, hidden_size]
        # 基础 RNN 是单向的，num_directions=1
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 步骤2：前向传播通过 RNN 层
        # 输出 outputs 和 final_hidden
        # outputs：形状 [batch_size, sequence_length, hidden_size]（每个时间步的隐藏状态）
        # final_hidden：形状 [num_layers, batch_size, hidden_size]（最后一个时间步的隐藏状态）
        outputs, final_hidden = self.rnn(x, h0)

        # 步骤3：处理输出（根据任务选择：用最后一个时间步的隐藏状态做预测）
        # 由于 batch_first=True，outputs 的第 2 维是序列长度，取 [-1] 即最后一个时间步
        out = self.fc(outputs[:, -1, :])

        return out


# 3.初始化模型、损失函数、优化器
model = SimpleRNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()        # 均方误差损失（回归任务）
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 4.构造模拟序列数据（输入序列 + 真实标签）
# 输入 x 形状：[batch_size, sequence_length, input_size]（符合 batch_first=True）
x = torch.randn(batch_size, sequence_length, input_size, dtype=torch.float32)
# 真实标签 y 形状：[batch_size, output_size]
y_true = torch.randn(batch_size, output_size, dtype=torch.float32)


# 5. 单轮训练（前向传播 + 反向传播 + 参数更新）
print("===== 训练前，RNN 第一层权重（部分） =====")
print(model.rnn.weight_ih_l0[:2, :2])  # 查看 RNN 输入到隐藏层的权重（部分）

# 前向传播
y_pred = model(x)
loss = criterion(y_pred, y_true)
print(f"\n===== 前向传播完成，损失值：{loss.item():.4f} =====")

# 反向传播
optimizer.zero_grad()
loss.backward()
print("\n===== 反向传播完成，RNN 第一层权重梯度（部分） =====")
print(model.rnn.weight_ih_l0.grad[:2, :2])

# 参数更新
optimizer.step()
print("\n===== 参数更新完成，RNN 第一层权重（部分） =====")
print(model.rnn.weight_ih_l0[:2, :2])

