# 输入历史
user_input_history = []

while True:
    user_input = input("请输入：")
    user_input_history.append(user_input)

    if user_input == '退出' or user_input == 'exit' or user_input == 'EXIT':
        break


# 遍历输入历史
print('\n' * 3 + '=' * 30)
for i, ih in enumerate(user_input_history):
    print(f'输入{i + 1}：{ih}')

