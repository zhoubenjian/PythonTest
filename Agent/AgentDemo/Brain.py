import os
from openai import OpenAI
from dotenv import load_dotenv


# 加载.env 文件中的环境变量
load_dotenv()
# 这里设置你申请的 key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"

client = OpenAI(
    api_key = DEEPSEEK_API_KEY,
    base_url = DEEPSEEK_API_URL
)


'''
Agent 的大脑，负责思考与决策
'''
class AgentBrain:
    def __init__(self, model = 'deepseek-v4-flash'):
        self.model = model

    '''
    核心思考函数：接收提示，返回模型的思考结果
    '''
    def think(self, prompt):
        try:
            response = client.chat.completions.create(
                model = self.model,
                messages = [{"role": "user", "content": prompt}],
                # 控制创造性，越低越专注
                temperature = 0.5,
                # 控制回复长度
                max_tokens = 500
            )
            # 提取模型返回的文本内容（v1.x 版本属性路径变更）
            reasoning = response.choices[0].message.content
            return reasoning.strip()
        except Exception as e:
            return f'思考过程出错：{e}'


# 简单测试一下大脑是否工作
if __name__ == '__main__':
    brain = AgentBrain()
    test_prompt = '你好，请介绍一下你自己。'
    print('测试提问：', test_prompt)
    print('大脑回复：', brain.think(test_prompt))
