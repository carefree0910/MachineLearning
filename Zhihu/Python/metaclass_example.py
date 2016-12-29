class Person:
    def __init__(self):
        self.ability = 1

    def eat(self):
        print("Eat: ", self.ability)

    def sleep(self):
        print("Sleep: ", self.ability)

    def save_life(self):
        print("+ ", self.ability, " s")


class Wang(Person):
    def eat(self):
        print("Eat: ", self.ability * 2)


class Zhang(Person):
    def sleep(self):
        print("Sleep: ", self.ability * 2)


class Jiang(Person):
    def save_life(self):
        print("+ inf s")


class Mixture(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        person1, person2, person3 = bases

        def eat(self):
            person1.eat(self)

        def sleep(self):
            person2.sleep(self)

        def save_life(self):
            person3.save_life(self)

        for key, value in locals().items():
            if str(value).find("function") >= 0:
                attr[key] = value

        return type(name, bases, attr)


class Compare(Zhang, Jiang, Wang):
    pass


class Hong(Wang, Zhang, Jiang, metaclass=Mixture):
    pass


def test(person):
    person.eat()
    person.sleep()
    person.save_life()

if __name__ == '__main__':
    test(Hong())
    test(Compare())
