
import random
import copy
import sys
import threading
import numpy as np
import matplotlib.pyplot as plt



(ALPHA, BETA, RHO, Q) = (1.0, 5.0, 0.5, 100)
(city_num, pso_num) = (101, 100)
data = np.fromfile('eil101', sep=' ')
data = data.reshape(city_num, 3)
data1 = np.delete(data, 0, axis=1)
datax = np.delete(data1, 1, axis=1)
distance_x = datax.reshape(city_num).tolist()
datay = np.delete(data1, 0, axis=1)
distance_y = datay.reshape(city_num).tolist()
distance_graph = [[0.0 for col in range(city_num)] for raw in range(city_num)]
viscous_graph = [[[0.5 for col in range(city_num)] for raw in range(city_num)] for pn in range(pso_num)]
class PSO():

    # 初始化
    def __init__(self, ID):
        self.ID = ID
        self.__clean_data()
    # 初始数据
    def __clean_data(self):

        self.path = []
        self.total_distance = 0.0
        self.p_fit = 1000000
        self.pbest = []
        self.move_count = 0  # 移动次数
        self.current_city = -1  # 当前停留的城市
        self.open_table_city = [True for i in range(city_num)]  # 探索城市的状态

        city_index = random.randint(0, city_num - 1)  # 随机初始出生点
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1

    # 选择下一个城市
    def __choice_next_city(self,pn):
        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  # 存储去下个城市的概率
        total_prob = 0.0
        # 获取去下一个城市的概率
        for i in range(city_num):
            if self.open_table_city[i]:
                try:
                    select_citys_prob[i] = pow(viscous_graph[pn][self.current_city][i], ALPHA) * pow(
                        (1 / distance_graph[self.current_city][i]), BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    pass

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
    def search_path(self,pn):
        # 初始化数据
        self.__clean_data()
        # 搜素路径，遍历完所有城市为止
        while self.move_count < city_num:
            # 移动到下一个城市
            next_city = self.__choice_next_city(pn)
            self.__move(next_city)
        # 计算路径总长度
        self.__cal_total_distance()

# ----------- TSP问题 -----------

class TSP(object):

    def __init__(self, n=city_num):
        # 城市数目初始化为city_num
        self.n = n
        self.__lock = threading.RLock()  # 线程锁
        self.x_fit = np.zeros(pso_num)
        self.new()
        self.x_axis = []
        self.y_axis = []
        # 计算城市之间的距离
        for i in range(city_num):
            for j in range(city_num):
                temp_distance = pow((distance_x[i] - distance_x[j]), 2) + pow((distance_y[i] - distance_y[j]), 2)
                temp_distance = pow(temp_distance, 0.5)
                distance_graph[i][j] = float(int(temp_distance + 0.5))
    def new(self, evt=None):
        # 停止线程



        self.psos = [PSO(ID) for ID in range(pso_num)]
        self.best_pso = PSO(-1)  # 初始最优解
        self.best_pso.total_distance = 1 << 31  # 初始最大距离
        self.iter = 1  # 初始化迭代次数
    def search_path(self,):
        self.iter = 1
        while self.iter <= 1000:
            '''
           global ALPHA
           ALPHA = 0.0015 * self.iter + 1
           global BETA
           BETA = 0.003 * self.iter + 4
           '''
            for pn in range(pso_num):
                # 搜索一条路径
                self.psos[pn].search_path(pn)
                if self.psos[pn].total_distance < self.psos[pn].p_fit:
                    self.psos[pn].p_fit = self.psos[pn].total_distance
                    self.psos[pn].pbest = self.psos[pn].path.copy()
                    if self.psos[pn].total_distance < self.best_pso.total_distance:
                        # 更新最优解
                        self.best_pso = copy.deepcopy(self.psos[pn])


            self.__update_viscous_gragh()
            self.x_axis.append(self.iter)
            self.y_axis.append(self.best_pso.total_distance)
            print("迭代次数为：", self.iter, "最佳路径总距离为：", self.best_pso.total_distance)
            print("最优解为：", self.best_pso.path)
            self.iter += 1

    def __update_viscous_gragh(self):

        temp_viscous = [[[0.0 for col in range(city_num)] for raw in range(city_num)] for pn in range(pso_num)]
        for pn in range(pso_num):

            decide = np.zeros((city_num, city_num))

            for i in range(1, city_num):
                start, end = self.psos[pn].path[i - 1], self.psos[pn].path[i]
                temp_viscous[pn][start][end] += 0.2
                decide[start][end] = 1
            temp_viscous[pn][self.psos[pn].path[city_num - 1]][self.psos[pn].path[0]] += 0.2
            decide[self.psos[pn].path[city_num - 1]][self.psos[pn].path[0]] = 1




            for i in range(1, city_num):
                start, end = self.psos[pn].pbest[i - 1], self.psos[pn].pbest[i]
                if decide[start][end] == 0:
                    temp_viscous[pn][start][end] += 0.35
            if decide[self.psos[pn].pbest[city_num - 1]][self.psos[pn].pbest[0]] == 0:
                temp_viscous[pn][self.psos[pn].pbest[city_num - 1]][self.psos[pn].pbest[0]] += 0.35

            for i in range(1, city_num):
                start, end = self.best_pso.path[i - 1], self.best_pso.path[i]
                if decide[start][end] == 0:
                    temp_viscous[pn][start][end] += 0.5
            if decide[self.best_pso.path[city_num-1]][self.best_pso.path[0]] == 0:
                temp_viscous[pn][self.best_pso.path[city_num-1]][self.best_pso.path[0]] += 0.5


        for pn in range(pso_num):
            for i in range(city_num):
                for j in range(city_num):
                    viscous_graph[pn][i][j] += 0.01
                    viscous_graph[pn][i][j] = viscous_graph[pn][i][j] * RHO + temp_viscous[pn][i][j]




if __name__ == '__main__':

    my_tsp = TSP()
    my_tsp.search_path()
    print(my_tsp.x_axis)
    print(my_tsp.y_axis)
    plt.figure(1)
    plt.title("Figure1")
    plt.xlabel("iters", size=14)
    plt.ylabel("fitness", size=14)
    # print(fitness)
    x = np.array(my_tsp.x_axis)
    y = np.array(my_tsp.y_axis)

    plt.plot(x, y, color='r', linewidth='1', linestyle='--')
    plt.xlabel("iters")
    plt.ylabel("distance")
    plt.text(1000, y[999], y[999])
    plt.grid(axis="x")
    plt.grid(axis="y")
    plt.title(label="test")
    plt.show()