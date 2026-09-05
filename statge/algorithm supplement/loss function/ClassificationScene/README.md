# 分类场景
    

# 二元分类交叉熵（Binary Cross-Entropy Loss）
    用于二分类任务，将输入映射到0到1之间的概率值，用于计算分类错误的损失；

    损失函数的定义为：Loss = -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
    直观理解:
        如果真实标签 y=1：损失为 -log(y_pred)，预测概率越接近 1 损失越小；
        如果真实标签 y=0：损失为 -log(1 - y_pred)，预测概率越接近 0 损失越小；

    ！！！
    关键规则（务必记住）
    在二元分类中，y_pred 永远是指"属于正类（类别 1）的概率"，而不是"属于类别 0 的概率"
    如果 y_pred = 0.9，意思是：P(类别=1) = 90%，P(类别=0) = 10%
    如果 y_pred = 0.2，意思是：P(类别=1) = 20%，P(类别=0) = 80%
    ！！！


    PyTorch 封装方法（最常用）
        二分类：
            方式一：BCEWithLogitsLoss(logits, y_true)（强烈推荐）直接使用    
                criterion = nn.BCEWithLogitsLoss()
                    输入：logits（未经过 Sigmoid 的原始输出），不是 y_pred（概率）
                    优点：内部做了数值稳定，不会出现 log(0) 或梯度爆炸
                    输出：标量损失值

            方式二：BCELoss（不推荐，极少场景）
                criterion = nn.BCELoss(y_pred, y_true)
                    输入：y_pred（概率），不是 logits（未经过 Sigmoid 的原始输出）
                    优点：内部不做数值稳定，会报错（y_pred 为 0 或 1）
                    输出：标量损失值
                    ⚠️ 警告：BCELoss 不会做数值稳定，如果 y_pred 恰好是 0 或 1，会报错！


    TensorFlow/Keras 封装方法
        二分类：
            方式一：直接使用（推荐）
                loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                loss_fn(y_true, logits)
                    from_logits=True：输入是 Logits（未经过 Sigmoid）
                    from_logits=False：输入是概率（已经过 Sigmoid）

            # 方式二：在编译模型时使用
                model.compile(
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    optimizer='adam'
                )


    Scikit-learn 封装方法
        二分类：
            方式一：直接使用（推荐）
                loss = log_loss(y_true, y_pred_proba)
                    优点：进行sigmoid处理后，内部做了数值稳定，不会出现 log(0) 或梯度爆炸
                    输出：标量损失值
    