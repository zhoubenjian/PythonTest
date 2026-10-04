'''
入门案例 3：循环图（你刚才写的 A/B 循环，反思 Agent 基础）
    Agent 调用工具 => 拿到结果，判断是否还要继续调用；满足条件才退出
'''
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    msg: str

def node_a(state: State):
    return {"msg": state["msg"] + " => A"}

def node_b(state: State):
    return {"msg": state["msg"] + " => B"}

# 路由：吐过包含exit就终止，否则继续去B
def route(state: State):
    if 'exit' in state['msg']:
        return 'END'
    else:
        return 'node_b'


builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)

builder.add_edge(START, "node_a")
builder.add_conditional_edges("node_a", route)
builder.add_edge("node_b", "node_a") # B执行完回到A，形成循环

graph = builder.compile()

# 测试：循环几次后，在msg里带上exit退出
# 注意坑：exit标记出现的那一轮A节点依然执行，之后才终止
res = graph.invoke({"msg": "start, exit"})
print(res)
