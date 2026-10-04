'''
入门案例 2：带条件分支（条件边，Agent 最常用）
    场景：输入文字，如果包含`问题`走 LLM 回答节点；否则走闲聊节点
'''
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. 定义状态
class State(TypedDict):
    text: str
    reply: str


# 2. 定义节点
# 节点1：回答问题
def answer_question(state: State):
    return {"reply": f"收到问题：{state['text']}，正在解答"}

# 节点2：闲聊
def chat_normal(state: State):
    return {"reply": f"闲聊：{state['text']}"}

# 路由函数
def route_func(state: State):
    if "问题" in state["text"]:
        return "answer_node"
    else:
        return "chat_node"


# 3. 构图
builder = StateGraph(State)
builder.add_node("answer_node", answer_question)
builder.add_node("chat_node", chat_normal)

# START 之后直接条件分流，避免先进入某个节点再判断导致循环
builder.add_conditional_edges(
    START,
    route_func,
    {
        "answer_node": "answer_node",
        "chat_node": "chat_node",
    }
)
builder.add_edge("answer_node", END)
builder.add_edge("chat_node", END)

graph = builder.compile()

print(graph.invoke({"text": "今天天气怎么样？"}))
print(graph.invoke({"text": "我有一个问题，什么是LangGraph?"}))