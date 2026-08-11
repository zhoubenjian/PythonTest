'''
langchain-deepseek直接调用DeepSeek（更语义化，可能支持更多 DeepSeek 专属特性）
    temperature：
        0.0	确定性输出，每次回答几乎一样	数学计算、代码生成、事实问答
        0.3-0.5	较低随机性，回答较保守	翻译、摘要、文档处理
        0.7-0.9	中等创造性，回答多样化	一般对话、创意写作、头脑风暴
        1.0+	高随机性，可能天马行空	诗歌创作、故事生成、角色扮演

    max_tokens：
        限制模型输出的最大长度

    timeout：
        请求超时时间

    max_retries：
        请求失败时的最大重试次数
'''
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek


# 加载环境变量
load_dotenv()


# 1.获取API KEY
API_KEY = os.getenv("DEEPSEEK_API_KEY")


# 2.模型初始化
ds_llm = ChatDeepSeek(
    api_key = API_KEY,
    model = "deepseek-v4-flash",
    temperature = 0,
    max_tokens = 1024,
    timeout = None,
    max_retries = 2
)


# 调用模型
response = ds_llm.invoke("你好，请介绍 LangChain")

print(response.content)