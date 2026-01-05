'''
装饰器
'''


class Product:
    def __init__(self, price):
        self.__price = price  # 使用私有属性作为内部存储

    # getter方法，获取价格
    @property
    def price(self):
        return self.__price

    # setter方法，赋值价格，并校验
    @price.setter
    def price(self, value):
        if value <= 0.0:
            # 抛出异常
            raise ValueError('Price cannot be negative!')
        self.__price = value


p = Product(99.9)
# 调用getter方法
print(p.price)  # 99.9

# 调用setter方法
p.price = 19.9
print(p.price)  # 19.9

# 抛出 ValueError
p.price = -0.9
