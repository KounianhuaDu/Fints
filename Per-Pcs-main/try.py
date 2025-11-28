class Node:
    __tablename__ = "table"
    name = "xiaoming"
    age = 14
    address = None
    cla = None
    grade = None

    def __init__(self, name, age, address=None, cla=None, grade=None):
        pass

    def print_attribute(self):
        s = self.__init__.__code__
        print(s.co_varnames)


print(Node.__init__.__code__.co_varnames)
print('self' in Node.__init__.__code__.co_varnames)
