import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math
import sys
start = time.perf_counter()


class TSP():

    def __init__(self, x_local, y_local):
        self.x_local = np.array(x_local)
        self.y_local = np.array(y_local)


class PSO():
    def __init__(self, pn, max_iter):

        self.tsp = TSP(x_local, y_local)
        #
        self.pn = pn  # 粒子数量
        self.dim = self.tsp.x_local.size  # 搜索维度
        self.max_iter = max_iter  # 迭代次数

        # 初始化参数设置
        self.ustks_u = 10 * self.max_iter / 100
        self.ustks_l = 1 * self.max_iter / 100
        self.alpha = 2
        self.i1_u = 10 / self.dim
        self.i1_l = 0 / self.dim

        # self.value = np.array(value)
        self.distance_graph = np.empty((self.dim, self.dim))
        self.c_distance_graph()
        self.neighbor_distance = np.argsort(self.distance_graph)
        #print(self.c_distance_graph())
        self.X0 = np.array([0])
        self.X_before = np.empty((self.pn, self.dim), dtype=int)
        self.X = np.empty((self.pn, self.dim), dtype=int)
        self.V = np.zeros((self.pn, self.dim, self.dim))  # 所有粒子的位置和速度，，可随机
        '''
        row_rand = random.sample(range(self.dim - 1), self.dim - 1)
        for i in range(self.pn):
            for row in row_rand:
                column = random.randint(0, self.dim - 1)
                self.V[i][row][column] = random.random()
            # print(self.V)
            self.V[i] += self.V[i].T - np.diag(self.V[i].diagonal())
        '''

        self.x_fit = np.zeros(self.pn)
        self.pbest = np.empty((self.pn, self.dim), dtype=int)  # 个体经历的最佳位置和全局最佳位置,pn*dim 低维重量
        self.gbest = np.empty((self.dim), dtype=int)
        self.p_fit = np.zeros(self.pn)  # 每个个体的历史最佳适应值
        self.fit = 100000000  # 全局最佳适应值
        self.stick = np.ones((self.pn, self.dim))
        #print(self.X0)

        # 初始化粒子群X,V,F
        for i in range(self.pn):

            self.X[i] = np.append(self.X0, random.sample(range(1, self.dim), self.dim-1))
            self.pbest[i] = self.X[i].copy()
            self.x_fit[i] = self.function(self.X[i])
            self.p_fit[i] = self.x_fit[i].copy()
            # 这个个体历史最佳的位置
            if self.x_fit[i] < self.fit:  # 得到现在最优和历史最优比较大小，如果现在最优大于历史最优，则更新历史最优
                self.fit = self.x_fit[i]
                self.gbest = self.X[i].copy()

    def c_distance_graph(self):
        for i in range(self.dim):
            for j in range(self.dim):
                self.distance_graph[i][j] = math.sqrt(pow((self.tsp.x_local[i] - self.tsp.x_local[j]), 2)
                                                      + pow((self.tsp.y_local[i] - self.tsp.y_local[j]), 2))
                if i == j:
                    self.distance_graph[i][j] = 10000


        return self.distance_graph




    def function(self, x):
        temp = 0
        for j in range(self.dim - 1):
            temp += self.distance_graph[x[j]][x[j + 1]]
        temp += self.distance_graph[x[self.dim - 1]][x[0]]
        f = temp
        return f
    def choice_next_city(self,current_city,open_city):

        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  # 存储去下个城市的概率
        total_prob = 0.0

        # 获取去下一个城市的概率
        for i in range(len(open_city)):
            if open_city[i]:
                try:
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(self.V[i][current_city][i], 1) * pow(
                        (100 / self.distance_graph[current_city][i]), 1.5)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print('baocuo')
                    sys.exit(1)

        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(self.dim):
                if open_city[i]:
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
            while ((open_city[next_city]) == False):  # if==False,说明已经遍历过了
                next_city = random.randint(0, city_num - 1)

        # 返回下一个城市序号
        return next_city


    def update(self):

        fitness = []


        #print(self.X.shape)
        iter = 0
        for t in range(self.max_iter):

            # 参数更新
            iter += 1

            self.i1 = self.i1_u - (iter / self.max_iter) * (self.i1_u - self.i1_l)
            self.i2 = self.alpha * (1 - self.i1) / (self.alpha + 1)
            self.i3 = (1 - self.i1) / (self.alpha + 1)
            self.ustks = self.ustks_l + (iter / self.max_iter) * (self.ustks_u - self.ustks_l)

            for i in range(self.pn):
                # 更新gbest\pbest

                self.x_fit[i] = self.function(self.X[i])

                if self.x_fit[i] < self.p_fit[i]:  # 更新个体最优
                    self.p_fit[i] = self.x_fit[i].copy()
                    self.pbest[i] = self.X[i].copy()  # .copy()

            p_id = np.argmin(self.p_fit)
            if self.p_fit[p_id] < self.fit:
                self.gbest = self.pbest[p_id].copy()
                self.fit = self.p_fit[p_id].copy()
            fitness.append(self.fit)

            print(iter)
            print(self.gbest)
            print(self.fit)
            #print(self.X)
            print(iter*0.35/self.max_iter+0.1)



            #学习部分

            #print(isupdate)


            for i in range(self.pn):


                open_city = [True for i in range(city_num)]
                xe = []
                pe = []
                ge = []
                p_x = []
                g_x = []
                # xe.append([self.X[i][0], self.X[i][1]])
                xe.append([self.X[i][0], self.X[i][self.dim - 1]])

                # pe.append([self.pbest[i][0], self.pbest[i][1]])
                pe.append([self.pbest[i][0], self.pbest[i][self.dim - 1]])

                # ge.append([self.gbest[0], self.gbest[1]])
                ge.append([self.gbest[0], self.gbest[self.dim - 1]])

                for j in range(self.dim - 1):
                    # if self.X[i][j] < self.X[i][j - 1]:
                    # xe.append([self.X[i][j], self.X[i][j - 1]])
                    # if self.X[i][j] < self.X[i][j + 1]:
                    xe.append([self.X[i][j], self.X[i][j + 1]])

                    # if self.pbest[i][j] < self.pbest[i][j - 1]:
                    # pe.append([self.pbest[i][j], self.pbest[i][j - 1]])
                    # if self.pbest[i][j] < self.pbest[i][j + 1]:
                    pe.append([self.pbest[i][j], self.pbest[i][j + 1]])

                    # if self.gbest[j] < self.gbest[j - 1]:
                    # ge.append([self.gbest[j], self.gbest[j - 1]])
                    # if self.gbest[j] < self.gbest[j + 1]:
                    ge.append([self.gbest[j], self.gbest[j + 1]])
                '''
                if self.X[i][self.dim - 1] < self.X[i][self.dim - 2]:
                    xe.append([self.X[i][self.dim - 1], self.X[i][self.dim - 2]])
                if self.pbest[i][self.dim - 1] < self.pbest[i][self.dim - 2]:
                    pe.append([self.pbest[i][self.dim - 1], self.pbest[i][self.dim - 2]])
                if self.gbest[self.dim - 1] < self.gbest[self.dim - 2]:
                    ge.append([self.gbest[self.dim - 1], self.gbest[self.dim - 2]])
                '''
                # print(pe)
                # print()
                xe_sort = np.sort(xe).tolist()
                pe_sort = np.sort(pe).tolist()
                ge_sort = np.sort(ge).tolist()
                for j in range(self.dim):
                    if pe_sort[j] not in xe_sort:
                        p_x.append(pe[j])
                    if ge_sort[j] not in xe_sort:
                        # print(22222222222)
                        g_x.append(ge[j])
                # print(g_x)
                l_p_x = len(p_x)
                l_g_x = len(g_x)
                temp = self.V[i].copy()
                self.V[i] = temp * 0.5
                #print(g_x)
                for k in range(len(xe)):
                    a = random.randint(0, self.dim-1)
                    self.V[i][xe[k][0]][a] = self.V[i][xe[k][0]][a] + 1500/26000
                for k in range(len(xe)):
                    self.V[i][xe[k][0]][xe[k][1]] = self.V[i][xe[k][0]][xe[k][1]] + 2000/self.x_fit[i]
                for k in range(l_p_x):
                    self.V[i][p_x[k][0]][p_x[k][1]] = self.V[i][p_x[k][0]][p_x[k][1]] + 3000/self.p_fit[i]
                for k in range(l_g_x):
                    self.V[i][g_x[k][0]][g_x[k][1]] = self.V[i][g_x[k][0]][g_x[k][1]] + 2000/self.fit
                # print(self.V[i][0])
                #self.V[i][self.V[i] < 0.000001] = 0
                # print(self.V[i].sum())
                current_city = 0
                new_X = np.array([current_city])
                #print(self.V[i])
                #V_copy = self.V[i].copy()
                #V_copy[V_copy < 0.2] = 0
                #V_copy[:, current_city] = 0
                open_city[current_city] = False

                # print(self.V[0][0])
                for h in range(self.dim - 1):
                    '''
                    if sum(V_copy[current_city]) != 0:
                        V_copy[current_city] = V_copy[current_city] * (1 / sum(V_copy[current_city]))
                        next_city = np.random.choice(city, 1, p=V_copy[current_city])[0]
                        #next_city = np.argmax(V_copy[current_city])
                        #print(next_city)
                        # print(next_city)
                    else:
                        r = random.random()
                        if r < 0.5:
                            next_city = np.random.choice(open_city, 1)[0]
                        else:
                            for nei in range(self.dim):
                                if self.neighbor_distance[current_city][nei] in open_city:
                                    next_city = self.neighbor_distance[current_city][nei]
                                    break
                    '''
                    next_city = self.choice_next_city(current_city, open_city)
                    # print(self.V[i][j])
                    new_X = np.append(new_X, next_city)
                    #V_copy[:, next_city] = 0
                    open_city[current_city] = False
                    current_city = next_city
                self.X[i] = new_X


           #先驱粒子
#########################################
            for k in range(1000):

                poineer = self.gbest.copy()
                r = random.random()
                # 三角形交换

                # 任意片段逆序
                if r < 0.85:
                    re_1 = random.randint(1, self.dim - 1)
                    re_2 = random.randint(1, self.dim - 1)
                    re1 = min(re_1, re_2)
                    re2 = max(re_1, re_2)
                    temp_re = poineer[re1:re2].copy()
                    poineer[re1:re2] = temp_re[::-1].copy()
                # 两交换
                if 0.85 <= r < 1:
                    ch1 = random.randint(1, self.dim - 1)
                    ch2 = random.randint(1, self.dim - 1)
                    poineer = np.insert(poineer, ch1, poineer[ch2])
                    if ch1 < ch2:
                        poineer = np.delete(poineer, ch2 + 1)
                    else:
                        poineer = np.delete(poineer, ch2)
                # 四边形交换


                poineer_f = self.function(poineer)
                if poineer_f < self.fit:
                    print('yes,gbest更新了+++++++++++++++++')
                    self.gbest = poineer.copy()
                    self.fit = poineer_f

                '''
                else:
                    det = poineer_f - self.fit
                    r = random.random()
                    p = math.exp(-det / T)
                    if p > r:
                        self.gbest = poineer.copy()
                        self.fit = poineer_f
                '''






            '''
            before_X = self.X
            # print(self.V)

            r = np.random.rand(self.pn, self.dim)
            c = np.greater(self.V, r)
            d = c.astype(int)
            for i in range(self.pn):
                count = 0
                for j in range(self.dim):
                    if d[i][j] == 1 and self.X[i][j] != self.gbest[j]:
                        count += 1
                        tempij = self.X[i][j]
                        same_id = np.argwhere(self.X[i] == self.gbest[j])
                        self.X[i][j] = self.gbest[j].copy()
                        self.X[i][same_id] = tempij
                        if count >= 5 :
                            break
                    if (self.X[i] == self.gbest).all:
                        c1 = random.randint(0,self.dim-1)
                        c2 = random.randint(0,self.dim-1)
                        temp_gj = self.gbest[c1]
                        self.gbest[c1] = self.gbest[c2]
                        self.gbest[c2] = temp_gj

            # 更新粘性
            decide = np.equal(before_X, self.X).astype(int)
            # print(decide)
            before_stick = self.stick
            self.stick = (1 - decide) * np.max(before_stick - 1 / self.ustks, 0)

            # print(decide)
        '''
        return fitness


'''
            if iter > 100 and (fitness[iter - 1] == fitness[iter - 50]) and iter%50==0:
                self.V = np.random.rand(self.pn, self.dim)
'''

city_num = 52
data = np.fromfile('bolin52', sep=' ')
data = data.reshape(city_num, 3)
data1 = np.delete(data, 0, axis=1)
datax = np.delete(data1, 1, axis=1)
x_local = datax.reshape(city_num)
datay = np.delete(data1, 0, axis=1)
y_local = datay.reshape(city_num)
my_tsp = TSP(x_local, y_local)

my_pso = PSO(pn=52, max_iter=1000)
fitness = my_pso.update()
'''
tt = []
for i in range(30):
    my_pso = PSO(pn=5, max_iter=1000)
    fitness = my_pso.update()
    tt.append(max(fitness))

print(f'历史最优值为：{max(tt)}')
print(f'均值为：{np.mean(tt)}')
print(f'标准差为：{np.std(tt)}')

end = time.perf_counter()
print('Running time: %s Seconds' % (end - start))
# print(my_bag.rate)
# 绘图
'''

plt.figure(1)
plt.title("Figure1")
plt.xlabel("iters", size=14)
plt.ylabel("fitness", size=14)
# print(fitness)
x = np.arange(0, my_pso.max_iter)
y = np.array(fitness)
plt.plot(x, y, color='b', linewidth='1', linestyle='-.', label='SBPSO')
plt.xlabel("iters")
plt.ylabel("profit")
plt.text(my_pso.max_iter, fitness[my_pso.max_iter - 1], fitness[my_pso.max_iter - 1])
plt.grid(axis="x")
plt.grid(axis="y")
plt.title(label="SBPSO")
#路径图
plt.show()
xx, yy = [], []
for i in range(city_num):
    xx.append(x_local[my_pso.gbest[i]])
    yy.append(y_local[my_pso.gbest[i]])
xx.append(xx[0])
yy.append(yy[0])
plt.plot(xx, yy, 'r-', linewidth=0.8)
plt.xlabel('xx')
plt.ylabel('yy')
plt.show()

'''
plt.figure(1)
plt.title("Figure1")
plt.xlabel("iters", size=14)
plt.ylabel("fitness", size=14)
# print(fitness)
x = np.arange(0,30, 1)
y = np.array(tt)
plt.plot(x, y, color='b',linewidth='1',linestyle='-.',label='SBPSO')
plt.xlabel("num")
plt.ylabel("profit")
#plt.text(my_pso.max_iter, fitness[my_pso.max_iter - 1], fitness[my_pso.max_iter - 1])
plt.grid(axis="x")
plt.grid(axis="y")
plt.title(label="test")
plt.show()
'''






