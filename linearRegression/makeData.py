
# coding:utf-8
import random

red_balls = list(range(1,34))

blue_balls = list(range(1,17))

def oneMat():
    
    red_list = random.sample(red_balls, 6) 

    blue_list = random.sample(blue_balls, 1)
    all_balls = sorted(red_list) + blue_list

    result = list(map(lambda a:str(a), all_balls))

    
    return "\t".join(result)


for i in range(10000):
    print(oneMat())
