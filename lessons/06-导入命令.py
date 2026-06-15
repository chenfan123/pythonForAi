# from 模块名 import 函数名
# from 模块名 import 函数名 as 别名
# from 模块名 import *
# import 模块名
# import 模块名 as 别名
# import 模块名.*

from math import cos, sin, pi, floor
from random import randint, sample
from statistics import mean, stdev  # 统计包

print(cos(pi/2))
print(sin(pi/2))
print(pi)
print(floor(1.23456789))  # 向下取整 1
print(randint(1, 10))  # 随机生成一个1到10之间的整数
print(mean([1, 2, 3, 4, 5]))  # 平均值 3
print(stdev([1, 2, 3, 4, 5]))  # 标准差 1.4142135623730951
print(sample([1, 2, 3, 4, 5], 3))  # 随机生成一个3个元素的列表
