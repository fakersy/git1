# -*- coding: utf-8 -*-
import random
import copy
import time
import sys
import math
import tkinter  # //GUI模块
import threading
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

# 参数
'''
ALPHA:信息启发因子，值越大，则蚂蚁选择之前走过的路径可能性就越大
      ，值越小，则蚁群搜索范围就会减少，容易陷入局部最优
BETA:Beta值越大，蚁群越就容易选择局部较短路径，这时算法收敛速度会
     加快，但是随机性不高，容易得到局部、的相对最优
'''
(ALPHA, BETA, RHO, Q) = (1.0, 2.0, 0.5, 100)
# 城市数，蚁群
(city_num, ant_num) = (198, 100)
data = np.fromfile('d198', sep=' ')
data = data.reshape(city_num, 3)
data1 = np.delete(data, 0, axis=1)
datax = np.delete(data1, 1, axis=1)
distance_x = datax.reshape(city_num).tolist()
datay = np.delete(data1, 0, axis=1)
distance_y = datay.reshape(city_num).tolist()
# 城市距离和信息素
distance_graph = [[0.0 for col in range(city_num)] for raw in range(city_num)]
pheromone_graph = [[1.0 for col in range(city_num)] for raw in range(city_num)]


# ----------- 蚂蚁 -----------
class Ant(object):

    # 初始化
    def __init__(self, ID):

        self.ID = ID  # ID
        self.__clean_data()  # 随机初始化出生点

    # 初始数据
    def __clean_data(self):

        self.path = []  # 当前蚂蚁的路径
        self.total_distance = 0.0  # 当前路径的总距离
        self.move_count = 0  # 移动次数
        self.current_city = -1  # 当前停留的城市
        self.open_table_city = [True for i in range(city_num)]  # 探索城市的状态

        city_index = random.randint(0, city_num - 1)  # 随机初始出生点
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1

    # 选择下一个城市
    def __choice_next_city(self):

        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  # 存储去下个城市的概率
        total_prob = 0.0

        # 获取去下一个城市的概率
        for i in range(city_num):
            if self.open_table_city[i]:
                try:
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow(
                        (1.0 / distance_graph[self.current_city][i]), BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID=self.ID,
                                                                                                current=self.current_city,
                                                                                                target=i))
                    sys.exit(1)

        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(city_num):
                if self.open_table_city[i]:
                    # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break

        # 未从概率产生，顺序选择一个未访问城市
        # if next_city == -1:
        #     for i in range(city_num):
        #         if self.open_table_city[i]:
        #             next_city = i
        #             break

        if (next_city == -1):
            next_city = random.randint(0, city_num - 1)
            while ((self.open_table_city[next_city]) == False):  # if==False,说明已经遍历过了
                next_city = random.randint(0, city_num - 1)

        # 返回下一个城市序号
        return next_city

    # 计算路径总距离
    def __cal_total_distance(self):

        temp_distance = 0.0

        for i in range(1, city_num):
            start, end = self.path[i], self.path[i - 1]
            temp_distance += distance_graph[start][end]

        # 回路
        end = self.path[0]
        temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance

    # 移动操作
    def __move(self, next_city):

        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.total_distance += distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1

    # 搜索路径
    def search_path(self):

        # 初始化数据
        self.__clean_data()

        # 搜素路径，遍历完所有城市为止
        while self.move_count < city_num:
            # 移动到下一个城市
            next_city = self.__choice_next_city()
            self.__move(next_city)

        # 计算路径总长度
        self.__cal_total_distance()


# ----------- TSP问题 -----------

class TSP(object):

    def __init__(self, n=city_num):

        # 创建画布

        # 城市数目初始化为city_num
        self.n = n
        # tkinter.Canvas



        self.__lock = threading.RLock()  # 线程锁


        self.new()
        self.x_axis = []
        self.y_axis = []


        # 计算城市之间的距离
        for i in range(city_num):
            for j in range(city_num):
                temp_distance = pow((distance_x[i] - distance_x[j]), 2) + pow((distance_y[i] - distance_y[j]), 2)
                temp_distance = pow(temp_distance, 0.5)
                distance_graph[i][j] = float(int(temp_distance + 0.5))

    # 按键响应程序


    # 更改标题


    # 初始化
    def new(self, evt=None):

        # 停止线程
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()





        # 初始城市之间的距离和信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = 1.0

        self.ants = [Ant(ID) for ID in range(ant_num)]  # 初始蚁群
        self.best_ant = Ant(-1)  # 初始最优解
        self.best_ant.total_distance = 1 << 31  # 初始最大距离
        self.iter = 1  # 初始化迭代次数

    # 将节点按order顺序连线


    # 清除画布


    # 退出程序

    # 停止搜索


    # 开始搜索
    def search_path(self,):
        self.iter = 0
        while self.iter <= 1000:


            # 遍历每一只蚂蚁
            for ant in self.ants:
                # 搜索一条路径
                ant.search_path()
                # 与当前最优蚂蚁比较
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)
            # 更新信息素
            self.x_axis.append(self.iter)
            self.y_axis.append(self.best_ant.total_distance)
            self.__update_pheromone_gragh()
            print(u"迭代次数：", self.iter, u"最佳路径总距离：", (self.best_ant.total_distance))
            print(u"最佳路径：", (self.best_ant.path))
            # 连线

            # 设置标题


            self.iter += 1

    # 更新信息素
    def __update_pheromone_gragh(self):

        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        for ant in self.ants:
            for i in range(1, city_num):
                start, end = ant.path[i - 1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += Q / ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]

        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j]

    # 主循环



# ----------- 程序的入口处 -----------
if __name__ == '__main__':
    tongji = []

    for i in range(10):
        my_tsp = TSP()
        my_tsp.search_path()
        tongji.append(min(my_tsp.y_axis))
    print(my_tsp.y_axis)
    print(min(tongji))
    print(sum(tongji) / 10)
    print(max(tongji))
    print(tongji)

    # my_tsp = TSP()
    # my_tsp.search_path()
    # print(my_tsp.x_axis)
    # print(my_tsp.y_axis)
    # plt.figure(1)
    # plt.title("Figure1")
    # plt.xlabel("iters", size=14)
    # plt.ylabel("fitness", size=14)
    # # print(fitness)
    # x = np.array(my_tsp.x_axis)
    # y = np.array(my_tsp.y_axis)
    # print(x)
    # print(y)
    # plt.plot(x, y, color='r', linewidth='1', linestyle='--')
    # plt.xlabel("iters")
    # plt.ylabel("distance")
    # plt.text(1000, y[999], y[999])
    # plt.grid(axis="x")
    # plt.grid(axis="y")
    # plt.title(label="test")
    # plt.show()