import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 加载环境变量
load_dotenv()


# 1.系统提示
prompt = ChatPromptTemplate.from_template(
    "请回答：{question}"
)


# 2.初始化模型
model = ChatOpenAI(
    model = "deepseek-v4-flash",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com/v1"
)


# 3.创建输出解析器，提取纯文本
output_parser = StrOutputParser()


# 4.创建 Chain
chain = prompt | model | output_parser


# 5.调用，执行
result = chain.invoke({
    'question': '预估中国2026年经济增长率范围'
})


# 6.打印回答内容
print(result)







