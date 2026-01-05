'''
多继承
'''


# 父类
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print(f'{self.name} make a sound.')

# 父类
class Flyable:
    def fly(self):
        print('Flying...')

# 父类
class Swimmable:
    def swim(self):
        print('Swimming...')

# 子类
class Duck(Animal, Flyable, Swimmable):
    def speak(self):
        print('Quack！')



duck = Duck("Donald")
duck.speak()    # Quack！
duck.fly()      # Flying...
duck.swim()     # Swimming...

