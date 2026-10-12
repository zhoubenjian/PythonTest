'''
入门案例 1：最简单线性图（无分支）
'''
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. 定义状态
class State(TypedDict):
    content: str


# 2. 定义节点
def node_1(state: State):
    return {"content": state["content"] + "【节点1处理】"}

def node_2(state: State):
    return {"content": state["content"] + "【节点2处理】"}


# 3. 构图
builder = StateGraph(State)
builder.add_node("n1", node_1)
builder.add_node("n2", node_2)

# Start -> n1 -> n2 -> End
builder.add_edge(START, "n1")
builder.add_edge("n1", "n2")
builder.add_edge("n2", END)

# 编译
graph = builder.compile()

# 调用 invoke
result = graph.invoke({"content": "你好"})
print(result)       # {'content': '你好【节点1处理】【节点2处理】'}
