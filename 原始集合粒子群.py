import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math

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
        # print(self.c_distance_graph())
        self.X0 = np.array([0])
        self.X = np.empty((self.pn, self.dim), dtype=int)
        self.V = np.random.rand(self.pn, self.dim)  # 所有粒子的位置和速度，，可随机
        self.pv = np.random.rand(self.pn, self.dim)
        self.gv = np.random.rand(self.pn, self.dim)
        self.pid = np.random.rand(self.pn, self.dim)
        self.gid = np.random.rand(self.pn, self.dim)
        self.x_fit = np.zeros(self.pn)
        self.pbest = np.empty((self.pn, self.dim), dtype=int)  # 个体经历的最佳位置和全局最佳位置,pn*dim 低维重量
        self.gbest = np.empty((self.dim), dtype=int)
        self.p_fit = np.zeros(self.pn)  # 每个个体的历史最佳适应值
        self.fit = 100000000  # 全局最佳适应值
        self.stick = np.ones((self.pn, self.dim))
        # print(self.X0)

        # 初始化粒子群X,V,F
        for i in range(self.pn):

            self.X[i] = np.append(self.X0, random.sample(range(1, self.dim), self.dim - 1))
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
        return self.distance_graph

    def function(self, x):
        temp = 0
        for j in range(self.dim - 1):
            temp += self.distance_graph[x[j]][x[j + 1]]
        temp += self.distance_graph[x[self.dim - 1]][x[0]]
        f = temp
        return f

    def update(self):

        fitness = []


        # print(self.X.shape)
        iter = 0
        for t in range(self.max_iter):

            # 参数更新
            iter += 1



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
            print(iter * 0.35 / self.max_iter + 0.1)

            detXG_x_local = np.empty((self.dim))
            detXG_y_local = np.empty((self.dim))
            detXP_x_local = np.empty((self.dim))
            detXP_y_local = np.empty((self.dim))
            detx = np.empty((self.dim))
            dety = np.empty((self.dim))
            for i in range(self.pn):
                for j in range(self.dim):
                    detXG_x_local[j] = self.tsp.x_local[self.gbest[j]] - self.tsp.x_local[self.X[i][j]]
                    detXG_y_local[j] = self.tsp.y_local[self.gbest[j]] - self.tsp.y_local[self.X[i][j]]
                    detXP_x_local[j] = self.tsp.x_local[self.pbest[i][j]] - self.tsp.x_local[self.X[i][j]]
                    detXP_y_local[j] = self.tsp.y_local[self.pbest[i][j]] - self.tsp.y_local[self.X[i][j]]
                    detx = detXG_x_local[j]+detXP_x_local[j]
                    dety = detXG_y_local[j]+detXP_y_local[j]







                X_copy = self.X[i].copy()

                point_1 = random.randint(1, self.dim - 1)
                # point2 = random.randint(1, self.dim - 1)
                point_2 = point_1 + 3
                fragment_1 = self.pbest[i][point_1:point_2].copy()
                fragment_2 = self.gbest[point_1:point_2].copy()
                r_ = random.random()
                if r_ > 0.7 + iter * 0.3 / self.max_iter:
                    # 取pbest片段
                    for j in range(len(fragment_1)):
                        temp = X_copy[point_1 + j]
                        X_copy[np.argwhere(X_copy == fragment_1[j])[0]] = temp
                        X_copy[point_1 + j] = fragment_1[j]
                    temp_x_fit = self.function(X_copy)
                    if temp_x_fit < self.x_fit[i]:
                        # print('yes,x更新了+++++++++++++++++')
                        self.x_fit[i] = temp_x_fit
                        self.X[i] = X_copy.copy()
                else:
                    # 取gbest片段
                    for j in range(len(fragment_2)):
                        temp = X_copy[point_1 + j]
                        X_copy[np.argwhere(X_copy == fragment_2[j])[0]] = temp
                        X_copy[point_1 + j] = fragment_2[j]

                    temp_x_fit = self.function(X_copy)
                    if temp_x_fit < self.x_fit[i]:
                        # print('yes,x更新了+++++++++++++++++')
                        self.x_fit[i] = temp_x_fit
                        self.X[i] = X_copy.copy()

                # 近邻表
                ''' for i in range(self.pn):
                neighbor_distance_copy = self.neighbor_distance.copy()
                X_copy = self.X[i].copy()
                for j in range(self.dim-1):
                    r = random.random()
                    if r < 0.6:
                        k = random.randint(0, 5)
                        X_copy[np.argwhere(X_copy == neighbor_distance_copy[j][k])[0][0]] = X_copy[j+1]
                        X_copy[j+1] = neighbor_distance_copy[j][k]
                        neighbor_distance_copy = np.delete(neighbor_distance_copy, k, axis=1)
                    temp_x_fit = self.function(X_copy)
                    if temp_x_fit < self.x_fit[i]:
                        self.x_fit[i] = temp_x_fit
                        self.X[i] = X_copy.copy()
                        if temp_x_fit < self.p_fit[i]:
                            self.p_fit[i] = temp_x_fit
                            self.pbest[i] = X_copy.copy()
                            if temp_x_fit < self.fit:
                                self.fit = temp_x_fit
                                self.gbest = X_copy.copy()

                '''
                # 学习2
                '''
                for i in range(self.pn):
                    if sum(abs(isupdate[i])) == 0:
                        re_1 = random.randint(1, self.dim - 1)
                        re_2 = random.randint(1, self.dim - 1)
                        re1 = min(re_1, re_2)
                        re2 = max(re_1, re_2)
                        for j in range(re1, int(re2 / 2 + re1 / 2)):
                            g_xj = np.argwhere(self.gbest == self.X[i][j])[0][0]
                            if g_xj == self.dim - 1:
                                g_xj = -1
                            self.X[i][np.argwhere(self.X[i] == self.gbest[g_xj + 1])] = self.X[i][j + 1]
                            self.X[i][j + 1] = self.gbest[g_xj + 1]
                        for j in range(int(re2 / 2 + re1 / 2), re2):
                            p_xj = np.argwhere(self.pbest[i] == self.X[i][j])[0][0]
                            if p_xj == self.dim - 1:
                                p_xj = -1
                            self.X[i][np.argwhere(self.X[i] == self.pbest[i][p_xj + 1])] = self.X[i][j + 1]
                            self.X[i][j + 1] = self.pbest[i][p_xj + 1]
                            for i in range(self.pn):
                for j in range(self.dim - 1):
                    g_xj = np.argwhere(self.gbest == self.X[i][j])[0][0]
                    p_xj = np.argwhere(self.pbest[i] == self.X[i][j])[0][0]
                    if g_xj == self.dim - 1:
                        g_xj = -1
                    if p_xj == self.dim - 1:
                        p_xj = -1
                    x_g = np.argwhere(self.X[i] == self.gbest[g_xj + 1])
                    x_p = np.argwhere(self.X[i] == self.pbest[i][p_xj + 1])
                    r_study = random.random()
                    if r_study < 0.15:
                        self.X[i][x_g] = self.X[i][j + 1]
                        self.X[i][j + 1] = self.gbest[g_xj + 1]
                    if r_study > 0.85:
                        self.X[i][x_p] = self.X[i][j + 1]
                        self.X[i][j + 1] = self.pbest[i][p_xj + 1]
                zuobiao
                [(565.0 ,575.0), (25.0 ,185.0), (345.0 ,750.0), (945.0 ,685.0), (845.0 ,655.0), (880.0 ,660.0), (25.0 ,230.0), (525.0, 1000.0), (580.0, 1175.0),
                       (650.0, 1130.0), (1605.0, 620.0), (1220.0, 580.0), (1465.0 ,200.0), (1530.0, 5.0), (845.0 ,680.0), (725.0, 370.0), (145.0 ,665.0), (415.0 ,635.0),
                       (510.0, 875.0), (560.0 ,365.0),(300.0 ,465.0),(520.0, 585.0),(480.0 ,415.0),(835.0 ,625.0),(975.0 ,580.0),(1215.0,245.0),(1320.0 ,315.0),(1250.0, 400.0),
                       (660.0 ,180.0),(410.0 ,250.0),(420.0, 555.0),(575.0, 665.0),(1150.0 ,1160.0),(700.0 ,580.0),(685.0 ,595.0),(685.0 ,610.0),(770.0 ,610.0),(795.0, 645.0),
                       (720.0, 635.0),(760.0 ,650.0),(475.0 ,960.0),(95.0 ,260.0),(875.0 ,920.0),(700.0 ,500.0),(555.0, 815.0),(830.0 ,485.0),(1170.0 ,65.0),(830.0 ,610.0),(605.0 ,625.0),
                       (595.0 ,360.0),(1340.0 ,725.0),(1740.0 ,245.0)]
                '''

            # 先驱粒子
            #########################################

            for j in range(self.dim - 4):
                poineer = self.gbest.copy()
                r = random.random()
                # 三角形交换
                if r < (iter * 0.35 / self.max_iter + 0.1):
                    triangle_1 = j
                    triangle_2 = triangle_1 + 1
                    triangle_3 = triangle_1 + 2
                    A = poineer[triangle_1]
                    B = poineer[triangle_2]
                    C = poineer[triangle_3]
                    ABC = self.distance_graph[A][B] + self.distance_graph[B][C]
                    ACB = self.distance_graph[A][C] + self.distance_graph[C][B]
                    BAC = self.distance_graph[B][A] + self.distance_graph[A][C]
                    if BAC < ACB < ABC:
                        poineer[triangle_2] = A
                        poineer[triangle_1] = B
                    if ACB < BAC < ABC:
                        poineer[triangle_2] = C
                        poineer[triangle_3] = B
                # 任意片段逆序
                if iter * 0.35 / self.max_iter + 0.1 <= r < 0.5:
                    re_1 = j
                    re_2 = random.randint(1, self.dim - 1)
                    re1 = min(re_1, re_2)
                    re2 = max(re_1, re_2)
                    temp_re = poineer[re1:re2].copy()
                    poineer[re1:re2] = temp_re[::-1].copy()
                # 两交换
                if 0.5 <= r < 1 - (iter * 0.35 / self.max_iter + 0.1):
                    ch1 = j
                    ch2 = random.randint(1, self.dim - 1)
                    temp_ch = poineer[ch1]
                    poineer[ch1] = poineer[ch2]
                    poineer[ch2] = temp_ch

                # 四边形交换
                if 1 - (iter * 0.35 / self.max_iter + 0.1) <= r < 1:
                    triangle_1 = j
                    triangle_2 = triangle_1 + 1
                    triangle_3 = triangle_1 + 2
                    triangle_4 = triangle_1 + 3
                    A = poineer[triangle_1]
                    B = poineer[triangle_2]
                    C = poineer[triangle_3]
                    D = poineer[triangle_4]
                    ABCD = self.distance_graph[A][B] + self.distance_graph[B][C] + self.distance_graph[C][D]
                    BCDA = self.distance_graph[B][C] + self.distance_graph[C][D] + self.distance_graph[D][A]
                    CDAB = self.distance_graph[C][D] + self.distance_graph[D][A] + self.distance_graph[A][B]
                    DABC = self.distance_graph[D][A] + self.distance_graph[A][B] + self.distance_graph[B][C]
                    if DABC < min(BCDA, CDAB) < ABCD:
                        poineer[triangle_1] = D
                        poineer[triangle_2] = A
                        poineer[triangle_3] = B
                        poineer[triangle_4] = C
                    if CDAB < min(DABC, BCDA) < ABCD:
                        poineer[triangle_1] = C
                        poineer[triangle_2] = D
                        poineer[triangle_3] = A
                        poineer[triangle_4] = B
                    if BCDA < min(DABC, CDAB) < ABCD:
                        poineer[triangle_1] = B
                        poineer[triangle_2] = C
                        poineer[triangle_3] = D
                        poineer[triangle_4] = A

                poineer_f = self.function(poineer)
                if poineer_f < self.fit:
                    print('yes,gbest更新了+++++++++++++++++')
                    self.gbest = poineer.copy()
                    self.fit = poineer_f

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

city_num = 101
data = np.fromfile('eil101', sep=' ')
data = data.reshape(city_num, 3)
data1 = np.delete(data, 0, axis=1)
datax = np.delete(data1, 1, axis=1)
x_local = datax.reshape(city_num)
datay = np.delete(data1, 0, axis=1)
y_local = datay.reshape(city_num)
my_tsp = TSP(x_local, y_local)

my_pso = PSO(pn=51, max_iter=1000)
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
# 路径图
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






