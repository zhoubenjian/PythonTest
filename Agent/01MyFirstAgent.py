# 1.定义工具
def get_weather(city):
    return f"{city}的天气是{city}的天气是晴朗的"


# 2.定义Agent核心循环
def run_agent(user_prompt):
    # 第一步：构建提示词，包含工具描述
    system_prompt = f"""
    你是一个助手。你可以使用以下工具：
    1. get_weather(city): 查询天气
    如果用户问天气，请返回 "ACTION: get_weather({{city}})"。
    否则直接回答。
    """

    # 第二步：调用LLM（这里假设你有一个llm.call 函数）
    # 实际开发中这里会调用 OpenAI/Anthropic API
    response = llm.call(system_prompt + user_prompt)

    # 第三步：判断是否需要行动
    if "ACTION" in response:
        # 解析工具调用（简单的字符串处理或正则）
        tool_name, args = parse_action(response)
        if tool_name == "get_weather":
            observation = get_weather(args)  # 执行工具
            # 第四步：把结果喂回给 LLM
            final_answer = llm.call(f"工具返回结果是：{observation}。请回答用户。")
            return final_answer
    else:
        return response


# 3.运行
print(run_agent("帮我查查北京的天气"))
