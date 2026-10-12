'''
LangGraph 实现简单翻译功能
    模仿 Agent/LangChain/03SimpleTranslate.py 的实现
    使用 LangGraph StateGraph 重构工作流
    langgraph + LLM(DeepSeek)
'''
import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 加载环境变量
load_dotenv()

# 配置
SOURCE_LANGUAGE = "中文"
TARGET_LANGUAGE = "English"


# ====== 1. 定义状态 ======
class TranslateState(TypedDict):
    '''
    翻译工作流的状态字典
    :param text: 待翻译的原文本
    :param source_language: 源语言
    :param target_language: 目标语言
    :param translated_text: 翻译后的文本
    '''
    text: str
    source_language: str
    target_language: str
    translated_text: str

# ====== 2. 初始化模型 ======
model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# ====== 3. 定义节点 ======
def translate_node(state: TranslateState) -> dict:
    '''
    翻译节点：调用 LLM 将文本从源语言翻译成目标语言
    :param state: 状态字典，包含 text, source_language, target_language
    :return: 包含 translated_text 的状态字典
    '''
    # 将 LangChain 的 ChatPromptTemplate 模式，改写为 SystemMessage + HumanMessage 组合
    system_prompt = SystemMessage(
        content=f"你是一个专业翻译，将{state['source_language']}翻译成{state['target_language']}。"
    )
    human_message = HumanMessage(content=state["text"])

    response = model.invoke([system_prompt, human_message])

    print(f"[translate_node] 翻译完成: {state['text']} => {response.content}")
    # 节点返回什么，就用什么覆盖掉同名字段，没返回的字段不动！
    return {"translated_text": response.content}

# ====== 4. 构建图 ======
builder = StateGraph(TranslateState)

# 添加节点
builder.add_node("translate", translate_node)

# 添加边（线性流转）
builder.add_edge(START, "translate")
builder.add_edge("translate", END)

# ====== 5. 编译图 ======
graph = builder.compile()

def do_translate(text: str, source_language: str = SOURCE_LANGUAGE, target_language: str = TARGET_LANGUAGE) -> str:
    '''
    封装的翻译函数，方便调用
    :param text: 待翻译文本
    :param source_language: 源语言
    :param target_language: 目标语言
    :return: 翻译后的文本
    '''
    result = graph.invoke({
        "text": text,
        "source_language": source_language,
        "target_language": target_language,
        "translated_text": ""  # 初始值，会被节点覆盖
    })
    return result["translated_text"]

# ====== 6. 运行 ======
if __name__ == "__main__":
    # ---------- 单次翻译测试 ----------
    print("=== 单次翻译测试 ===")
    result = do_translate("今天是10月12日", "中文", "英文")
    print(f"翻译结果: {result}\n")

    # ---------- 交互式翻译（模仿 LangChain 版本） ----------
    print("=== 交互式翻译（输入 '退出' 或 '结束' 停止）===")
    conversation_history = []

    while True:
        source_input = input(f"{SOURCE_LANGUAGE}: ")
        conversation_history.append(source_input)

        if "退出" in source_input or "结束" in source_input:
            break

        target_result = do_translate(source_input, SOURCE_LANGUAGE, TARGET_LANGUAGE)
        print(f"{TARGET_LANGUAGE}: {target_result}")
        conversation_history.append(target_result)

    # 打印历史记录
    print("\n" + "=" * 30)
    print("对话历史:")
    print("=" * 30)
    for i, ch in enumerate(conversation_history):
        print(ch)
        if (i + 1) % 2 == 0:
            print("-" * 30)
