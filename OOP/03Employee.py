# 父类
class Employee:
    def __init__(self, name, salary):
        self.name = name                # 公有属性(外部可访问和修改)
        self.__salary = salary          # __:私有属性(外部不可访问和修改)

    def get_info(self):
        return f'Name: {self.name}, Salary: {self.__salary}'


# 子类
class Manager(Employee):
    def __init__(self, name, salary, department):
        super().__init__(name, salary)  # 先调用父类方法
        self._department = department   # _:受保护的属性 (约定)(外部可访问和修改)

    def get_info(self):
        base_info = super().get_info()  # 先调用父类方法
        return f'{base_info}, Department: {self._department}'


manager = Manager('Alex', 18000, 'Software Engineer')
print(manager.get_info())

