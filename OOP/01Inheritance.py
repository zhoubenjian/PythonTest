'''
单继承
'''


# 父类
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print(f'{self.name} make a sound.')

# 继承类（狗）
class Dog(Animal):
    def speak(self):
        print(f'{self.name} barks!')

# 继承类（猫）
class Cat(Animal):
    def __init__(self, name, color):
        super().__init__(name)  # 调用父类的 __init__
        self.color = color

    def speak(self):
        super().speak()         # 先调用父类方法
        print(f'But {self.name} is a {self.color} cat and meows.')



dog = Dog('Buddy')
dog.speak()                     # Buddy barks!
print(isinstance(dog, Animal))  # True

print('-' * 30)

cat = Cat('Whiskers', "gray")
'''
Whiskers make a sound.
But Whiskers is a gray cat and meows.
'''
cat.speak()
print(isinstance(cat, Animal))  # True

