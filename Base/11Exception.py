'''
Exception(异常)：
    抛出异常（raise）和 return（返回）在本质上是互斥的，但在代码结构上又存在“替代”和“拦截”关系。
    一句话概括核心区别：return 属于正常流程，raise 属于异常流程；一旦 raise 执行，当前函数的 return 永远不会被执行。
'''


try:
    input_num = input('输入数字:')
    result = 100 / float(input_num)
    print(f'结果：{result}')

except ValueError:
    print('请输入有效数字！')

except ZeroDivisionError:
    print('不能除以零，输入不能为零！')

except Exception as e:
    print('未知错误：%s' % e)

else:
    # 可选
    print('执行时未发生异常！')

finally:
    # 可选
    print('无论如何都会执行，用于释放资源！')
