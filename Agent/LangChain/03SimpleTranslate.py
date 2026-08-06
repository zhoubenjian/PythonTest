'''
实现简单翻译功能
    langchain + LLM(DeepSeek)
'''
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# 加载环境变量
load_dotenv()

# 配置
source_language = "中文"
target_language = "英文"


# 1.系统提示模版
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业翻译，将{sources_language}翻译成{target_language}。"),
    ("human", "{text}")
])


# 2.初始化模型
model = ChatOpenAI(
    model = "deepseek-v4-flash",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com/v1"
)


# 3.创建输出解析器，提取纯文本
output_parser = StrOutputParser()


# 4.管道符（提示词 | 模型 | 输出）
chain = prompt | model | output_parser


# 5.执行链
translate_result = chain.invoke({
    "sources_language": "中文",
    "target_language": "英文",
    "text": "今天是8月6日"
})



if __name__ == "__main__":

    # 对话（翻译）历史
    conversation_history = []

    while True:
        source_language_input = input(f"{source_language}: ")
        conversation_history.append(source_language_input)

        if source_language_input == '退出' or source_language_input == '结束':
            break

        target_language_result = chain.invoke({
            "sources_language": f"{source_language}",
            "target_language": f"{target_language}",
            "text": f"{source_language_input}"
        })
        conversation_history.append(target_language_result)
        print(f'target language: {target_language_result}')


    # 历史记录
    print('\n' * 2)
    for i, ch in enumerate(conversation_history):
        print(ch)
        if (i + 1) % 2 == 0:
            print('=' * 30)


