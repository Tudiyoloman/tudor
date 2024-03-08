class Person(object):
    def __init__(self,name,age) -> None:
        self.name = name
        self.age = age

if __name__ == "__main__":
    person = Person("Tudor",20)
    print(person.age,person.name)
    

    