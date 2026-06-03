class Student:
    def __init__(self, no, name, gener, profession):
        self.__no = no
        self.__name = name
        self.__gender = gener
        self.__profession = profession


    '''
    静态方法（属于类）
    '''
    @staticmethod
    def intro():
        print('I graduated from Harvard University.')


    '''
    getter方法
    '''
    @property
    def no(self):
        return self.__no

    @property
    def name(self):
        return self.__name

    @property
    def gender(self):
        return self.__gender

    @property
    def profession(self):
        return self.__profession

    '''
    setter方法
    '''
    @no.setter
    def No(self, no):
        self.__no = no

    @name.setter
    def name(self, name):
        self.__name = name

    @gender.setter
    def gender(self, gender):
        self.__gender = gender

    @profession.setter
    def profession(self, profession):
        self.__profession = profession


Michael = Student(1001, 'Michael', 'Male', 'Software Engineering')
Alex = Student(1002, 'Alex', 'Male', 'Civil Engineering')

Student.intro()
Michael.intro()
Alex.intro()
