'''
抽象方法
'''

# 导入 ABC 和 abstractmethod
from abc import ABC, abstractmethod


# 继承 ABC 表示这是一个抽象基类（抽象类不能实例化！！！）
class Shape(ABC):
    @abstractmethod
    def area(self):
        # 计算面积，子类必须实现
        pass

    @abstractmethod
    def perimeter(self):
        # 计算周长，子类必须实现
        pass


class Rectangle(Shape):
    def __init__(self, width, height):
        self._width = width
        self._height = height

    def area(self):
        return self._width * self._height

    def perimeter(self):
        return 2 * (self._width + self._height)


rectangle = Rectangle(4.0, 5.0)
print('Area:', rectangle.area())  # 20.0
print('Perimater:', rectangle.perimeter())  # 18.0
