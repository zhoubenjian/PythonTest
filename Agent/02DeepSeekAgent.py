import os
from openai import OpenAI


# 初始化客户端
clinet = OpenAI(
    api_key = '',
    base_url = 'https://api.deepseek.com/v1'
)

# 调用对话API
try:
    response = clinet.chat.completions.create(
        model="deepseek-v4-pro",  # 指定模型，可选 deepseek-v4-flash / deepseek-v4-pro
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},  # 系统角色定义
            {"role": "user", "content": "Hello"},  # 用户提问
        ],
        stream=False  # 非流式输出（一次性返回完整结果）
    )
    # 打印回复内容
    print("回复结果：", response.choices[0].message.content)
except Exception as e:
    print("调用失败：", str(e))