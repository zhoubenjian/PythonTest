import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage


# 加载环境变量
load_dotenv()
# 这里设置你申请的 key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


llm = ChatOpenAI(
    model = "deepseek-chat",
    api_key = DEEPSEEK_API_KEY,    # 填写你的 API Key
    base_url = "https://api.deepseek.com/v1"    # 接口地址固定
)


def llm_mode(state: dict) -> dict:
    '''
    调用LLM大模型节点
    :param state:
    :return:
    '''
    system_prompt = SystemMessage(content="你是一个乐于助人的助手")

    # 将系统提示与对话历史合并
    messages = [system_prompt] + state["messages"]

    # 调用LLM
    response = llm.invoke(messages)

    return {"messages": [response]}


if __name__ == "__main__":
    res = llm_mode({"messages": [{"role": "user", "content": "你好"}]})
    print(res)
