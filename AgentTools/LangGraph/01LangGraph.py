'''
LangGraph
    LangGraph 是由 LangChain 团队开发的一个低层级 Agent 编排框架，专为构建有状态（Stateful）、长时运行的 AI 工作流而设计。
    与传统的线性 LLM 调用链不同，LangGraph 将工作流建模为有向图（Directed Graph）：
        ·节点（Node）：执行具体操作的函数（如调用 LLM、执行工具、处理数据）
        ·边（Edge）：定义节点之间的流转路径，支持条件分支
        ·状态（State）：在整个工作流中共享并传递的数据
'''
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# 1. 定义状态
class SimpleState(TypedDict):
    message: str
    processed: bool


# 2. 定义节点
def greet_node(state: SimpleState) -> dict:
    '''
    欢迎节点：生成问候语
    :param state: 状态字典，包含消息和 processed 标志
    :return: 包含问候语的状态字典
    '''
    print(f"[greet_node]收到消息: {state['message']}")
    return {"message": f"你好！{state['message']}"}

def process_node(state: SimpleState) -> dict:
    '''
    处理节点：对消息进行处理
    :param state: 状态字典，包含消息和 processed 标志
    :return: 状态字典，包含处理后的消息和 processed 标志
    '''
    print(f"[process_node]收到消息: {state['message']}")
    return {"processed": True}


# 3. 构建图
builder = StateGraph(SimpleState)

# 添加节点
builder.add_node("greet", greet_node)
builder.add_node("process", process_node)

# 添加边
builder.add_edge(START, "greet")
builder.add_edge("greet", "process")
builder.add_edge("process", END)


# 4. 编译图
graph = builder.compile()


# 5. 运行图
result = graph.invoke({
    "message": "你好",
    "processed": False,
})

print(f'\n运行结果: {result}')
