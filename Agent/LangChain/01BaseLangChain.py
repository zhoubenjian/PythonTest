import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()


# 1.初始化OpenAI模型
llm = ChatOpenAI(
    model = "deepseek-chat",
    api_key = "",    # 填写你的 API Key
    base_url = "https://api.deepseek.com/v1"            # 接口地址固定
)

# 2.提示词模版
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手"),
    ("user", "{input}")
])

# 3.输出解析器
parser = StrOutputParser()

# 4.链式组合（最简单的 LangChain 用法）（提示词 | 大模型 | 输出解析器）
chain = prompt | llm | parser

# 5.执行链
if __name__ == "__main__":
    result = chain.invoke({"input": "你好，介绍一下LangChain"})
    print(result)
