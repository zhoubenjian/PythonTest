'''
MNIST手写数字识别
    MNIST 数据集是一个用于训练和测试神经网络的数字识别数据集。
    它包含 60,000 张训练图像和 10,000 张测试图像，每个图像是一个手写数字的灰度图像。
'''
import tensorflow as tf
from tensorflow.keras import layers, models     # 导入 Keras 层模块和模型模块


'''
1.加载数据集
x_train 包含了 60000 张 28x28 像素的灰度图像
y_train 包含了对应的标签（0-9 的整数），表示图片上的数字是什么
'''
# 返回两个元组，分别包含训练集和测试集的图像和标签
(X_train_images, y_train_labels), (X_test_images, y_test_labels) = tf.keras.datasets.mnist.load_data()

'''
2.数据预处理
'''
# 图像数据归一化（将像素值从 0-255 缩放到 0-1 之间（除以 255.0））
X_train_images = X_train_images / 255.0
X_test_images = X_test_images / 255.0

'''
3.查看数据形状
'''
print(f"训练集图像形状: {X_train_images.shape}")       # 训练集图像形状: (60000, 28, 28)
print(f"训练集标签形状: {y_train_labels.shape}")       # 训练集标签形状: (60000,)


'''
4.构建 Sequential 顺序模型
①展平层（Flatten）：将 28x28 的二维图像“压扁”成一维的 784 维向量；
②全连接隐藏层（Dense）：包含 128 个神经元，使用 ReLU 激活函数，用于提取特征；
③全连接输出层（Dense）：包含 10 个神经元（对应 0-9 十个类别），使用 Softmax 激活函数，输出每个类别的概率；
'''
model = models.Sequential([
    # 展平层：将形状为 (28, 28) 的图像转换为形状为 (784,) 的一维向量
    # input_shape 告诉模型我们输入数据的形状（不包含 batch 维度）
    layers.Flatten(input_shape=(28, 28)),

    # 全连接隐藏层：128 个神经元，使用 ReLU 激活函数
    layers.Dense(128, activation='relu'),

    # 全连接输出层：10 个神经元，使用 Softmax 激活函数输出概率分布
    layers.Dense(10, activation='softmax'),
])

'''
4.打印模型结构，看看我们搭的“积木”长什么样
'''
model.summary()


'''
5.配置模型的训练参数
①损失函数（Loss Function）：用来衡量模型预测的概率分布与真实标签之间的差距；
②优化器（Optimizer）：也就是我们之前推导过的“梯度下降”算法的具体实现，它负责根据损失函数的梯度来更新模型参数（权重和偏置）;
③评估指标（Metrics）：用来在训练过程中直观地观察模型的表现（比如准确率）；
'''
model.compile(
    # 1.优化器：使用 Adam 优化器。它是梯度下降的一种高级自适应变体;
    # 能够自动调整学习率，在大多数深度学习任务中表现都非常出色。
    optimizer='adam',

    # 2.损失函数：因为这是一个多分类问题（10个类别），且输出层使用了 Softmax；
    # 所以我们选择稀疏分类交叉熵（Sparse Categorical Crossentropy）；
    # "Sparse" 是因为我们的标签 y_train 是整数（如 0, 1, 2），而不是 One-Hot 编码；
    loss='sparse_categorical_crossentropy',

    # 3.评估指标：我们选择准确率（Accuracy）在训练和测试过程中，监控模型的分类准确率
    metrics=['accuracy'],
)


'''
6.训练模型
'''
# epochs=5 表示将整个训练集完整地遍历 5 次
# batch_size=32 表示每次更新参数时，模型会看 32 张图片
history = model.fit(X_train_images, y_train_labels, epochs=5, batch_size=64)

# 评估模型
# 在训练结束后，用之前没见过的测试集来检验模型的泛化能力
print('--- 测试集评估结果 ---')
test_loss, test_acc = model.evaluate(X_test_images, y_test_labels)
print(f"测试集损失: {test_loss:.4f}")
print(f"测试集准确率: {test_acc:.4f}")
