from itertools import chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

memory = ChatMessageHistory()

chain_with_history = RunnableWithMessageHistory.from_history(
    chain,
    lambda x: memory,
    input_message_key="input",
    output_message_key="history"
)

# 能记住上下文
print(chain_with_history.invoke({"input": "我叫小明"}))
print(chain_with_history.invoke({"input": "我叫什么？"}))