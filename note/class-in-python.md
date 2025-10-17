# Python中类的继承

继承是 面向对象编程 (OOP) 的一个核心概念，它允许我们定义一个类（子类 或 派生类）来获取另一个类（父类、基类 或 超类）的所有属性和方法。这促进了代码的重用性和层次结构的建立。

## 1. 继承的基本概念
### 什么是继承？
简单来说，继承就是子类可以自动拥有父类的所有非私有成员（属性和方法）。子类可以在此基础上添加新的功能，或者重写（Override）父类的方法来实现自己的特定行为。

### 语法
在 Python 中，通过在子类定义时，将父类名称放在括号中来实现继承：

```Python

class ParentClass:
    # 父类的属性和方法
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        print(f"我是父类，我的名字是 {self.name}")

class ChildClass(ParentClass):
    # 子类的属性和方法
    def __init__(self, name, age):
        # 调用父类的构造方法初始化继承的属性
        super().__init__(name)
        self.age = age

    def introduce(self):
        print(f"我是子类，我叫 {self.name}，我今年 {self.age} 岁了。")

# 创建子类对象
c = ChildClass("小明", 10)
c.greet()      # 调用继承自父类的方法
c.introduce()  # 调用子类自己的方法
```
输出：
```bash
我是父类，我的名字是 小明
我是子类，我叫 小明，我今年 10 岁了。
```
## 2. super() 函数的使用
在子类中，如果我们需要调用（或引用）父类的方法，最常用的方式是使用 super() 函数。

### 调用父类的构造方法
这是 super() 最常见的用法。在子类的 __init__ 方法中调用父类的 __init__ 方法，可以确保父类中定义的属性得到正确初始化。

```Python

class Animal:
    def __init__(self, species):
        self.species = species
        print(f"一个 {self.species} 被创建了。")

class Dog(Animal):
    def __init__(self, name):
        # 使用 super() 调用父类 Animal 的 __init__ 方法
        super().__init__("狗") 
        self.name = name

d = Dog("旺财")
# 输出: 一个 狗 被创建了。
print(f"这个动物是 {d.species}，它的名字叫 {d.name}")
```
### 调用父类的普通方法
当子类重写了父类的方法，但又想在子类方法中执行父类的原始逻辑时，也可以使用 super()。

```Python

class Shape:
    def draw(self):
        print("正在绘制一个通用图形。")

class Circle(Shape):
    def draw(self):
        # 保持父类的功能
        super().draw() 
        # 添加子类特有的功能
        print("并将其定制为圆形。")

c = Circle()
c.draw()
```
输出：
```bash
正在绘制一个通用图形。
并将其定制为圆形。
```
## 3. 方法的重写 (Method Overriding)
方法重写是指子类定义了一个与父类中同名的方法。当子类的实例调用该方法时，Python 将执行子类中定义的新方法，而不是父类的原始方法。

```Python

class Vehicle:
    def move(self):
        print("交通工具正在以某种方式移动。")

class Car(Vehicle):
    # 重写 (Override) 了父类的 move 方法
    def move(self):
        print("汽车正在公路上行驶。")

class Boat(Vehicle):
    # 重写 (Override) 了父类的 move 方法
    def move(self):
        print("船只正在水面上航行。")

v = Vehicle()
c = Car()
b = Boat()

v.move() # 输出: 交通工具正在以某种方式移动。
c.move() # 输出: 汽车正在公路上行驶。
b.move() # 输出: 船只正在水面上航行。
```
这体现了 多态性 (Polymorphism)，即不同的对象对同一个方法调用产生不同的行为。

## 4. 继承的类型
### 单继承 (Single Inheritance)
一个子类只继承一个父类，这是最简单和常见的形式。

```Python

class A: pass
class B(A): pass  # B 继承 A
```
多重继承 (Multiple Inheritance)
一个子类可以同时继承多个父类。

```Python

class Flyable:
    def fly(self):
        print("我能飞。")

class Swimmable:
    def swim(self):
        print("我能游。")

class Duck(Flyable, Swimmable): # 鸭子同时继承了 Flyable 和 Swimmable
    pass

d = Duck()
d.fly()
d.swim()
```
📌 重点：方法解析顺序 (MRO)
在多重继承中，如果多个父类有同名的方法，Python 解释器需要一个规则来决定调用哪个父类的方法。这个规则被称为 方法解析顺序 (Method Resolution Order, MRO)。

Python 使用 C3 线性化算法来确定 MRO，其基本原则是 “从左到右，深度优先”。

可以通过子类的 __mro__ 属性或 help() 函数查看 MRO：

```Python

class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass

print(D.__mro__)
# 输出: (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)
```
这意味着当你调用 D 对象的方法时，搜索顺序是：D -> B -> C -> A -> object。

## 5. 继承与内置 object 类
在 Python 3 中，所有类都默认直接或间接继承自内置的 object 类。这是所有类的根基类。

```Python

class MyClass:
    pass # 等同于 class MyClass(object):

print(issubclass(MyClass, object)) # 输出: True
```
object 类提供了许多基本功能，例如 __init__、__str__、__delattr__ 等特殊方法，因此用户定义的所有类都天生具备这些能力。