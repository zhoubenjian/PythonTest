'''
模拟简易天气穿衣助手Agent（伪代码）
'''
import requests


class weather_agent:
    '''
    初始化
    '''
    def __init__(self):
        # 简单记忆存储
        self.memory = []
        # 工具集
        self.tools = {
            'get_weather': self.get_weather_api,
            'get_advice': self.get_advice_api
        }


    '''
    工具1：获取天气API
    '''
    def get_weather_api(self, city):
        """
        调用外部天气API获取数据
        """
        # 模拟调用天气API
        print(f"[Agent 行动] 正在查询{city}的天气...")

        # 假设返回的数据
        weather_data = {'city': city, 'temp': 29, 'condition': '晴朗', 'wind': '3级'}
        return weather_data


    '''
    工具2：根据天气生成建议
    '''
    def get_advice_api(self, weather_data):
        """
        根据天气数据生成穿衣建议
        """
        temp = weather_data['temp']
        condition = weather_data['condition']
        advice = f"当前{weather_data['city']}气温{temp}℃，天气{condition}。"
        if temp > 25:
            advice += '建议穿短袖、短裤。'
        elif temp > 15:
            advice += "建议穿长袖T恤、薄外套。"
        else:
            advice += "建议穿毛衣、厚外套。"
        return advice


    '''
    规划与执行核心
    '''
    def run(self, user_input):
        """
        解析用户目标并执行任务
        """
        print(f"[用户指令] {user_input}")

        # 步骤1: 规划 - 从指令中提取关键信息（城市）
        # 这里简化处理，实际会用更复杂的NLP模型
        if '天气' in user_input and '重庆' in user_input:
            city = '重庆'
        else:
            return "请告诉我您需要查询天气的城市。"

        # 步骤2: 行动 - 调用工具获取天气
        weather_data = self.tools['get_weather'](city)
        # 存入记忆
        self.memory.append({'step': 'fetched_weather', 'data': weather_data})

        # 步骤3: 行动 - 调用工具生成建议
        final_advice = self.tools['get_advice'](weather_data)
        # 存入记忆
        self.memory.append({'step': 'generated_advice', 'data': final_advice})

        return final_advice


# 使用Agent智能体
agent = weather_agent()
result = agent.run("重庆天气怎么样，如何穿搭？")
print(f"[Agent 回复] {result}")

