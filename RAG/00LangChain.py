'''
Implementing language translation with LangChain
'''
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# 加载环境变量
load_dotenv()


# 1.提示词模版
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的翻译官，请将以下文本从{source_language}翻译成{target_language}。"),
    ("human", "{text}")
])

# 2.初始化模型（OpenAI为例）
model = ChatOpenAI(
    model = "deepseek-v4-flash",    # deepseek-chat
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com/v1"
)

# 3.创建输出解析器，提取纯文本
output_parser = StrOutputParser()

# 4.管道符 | 构建链
chain = prompt | model | output_parser

# 5.执行链
result = chain.invoke({
    "source_language": "中文",
    "target_language": "英文",
    "text": "今天天气真好！"
})



if __name__ == "__main__":
    print(result)