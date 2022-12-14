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
        #print(self.c_distance_graph())
        self.X0 = np.array([0])
        self.X = np.empty((self.pn, self.dim), dtype=int)
        self.V = np.random.rand(self.pn,self.dim)  # 所有粒子的位置和速度，，可随机
        self.pv = np.random.rand(self.pn,self.dim)
        self.gv = np.random.rand(self.pn,self.dim)
        self.pid = np.random.rand(self.pn, self.dim)
        self.gid = np.random.rand(self.pn, self.dim)
        self.x_fit = np.zeros(self.pn)
        self.pbest = np.empty((self.pn, self.dim), dtype=int)  # 个体经历的最佳位置和全局最佳位置,pn*dim 低维重量
        self.gbest = np.empty((self.dim), dtype=int)
        self.p_fit = np.zeros(self.pn)  # 每个个体的历史最佳适应值
        self.fit = 100000  # 全局最佳适应值
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
        return self.distance_graph




    def function(self, x):
        temp = 0
        for j in range(self.dim - 1):
            temp += self.distance_graph[x[j]][x[j + 1]]
        temp += self.distance_graph[x[self.dim - 1]][x[0]]
        f = temp
        return f
    '''
    def X_reduce_X(self,x_1,x_2):
        point1 = random.randint(1, self.dim - 1)
        point2 = random.randint(1, self.dim - 1)
        point_1 = min(point1, point2)
        point_2 = max(point1, point2)
        fragment_1 = x_1[point_1:point_2].copy()
        fragment_2 = x_2[point_1:point_2].copy()
        r_ = random.random()
        if r_ > 0.5:
            #取gbest片段
            for j in range(len(fragment_1)):
                temp = self.X[i][point_1+j]
                self.X[i][point_1+j] = fragment_1[j]
                self.X[i][np.argwhere(self.X[i]==fragment_1[j])[0]] = temp
        else:
            for j in range(len(fragment_2)):
                temp = self.X[i][point_1+j]
                self.X[i][point_1+j] = fragment_2[j]
                self.X[i][np.argwhere(self.X[i]==fragment_2[j])[0]] = temp
        '''





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
            '''
            可能pfit,fit 未更新
            g_best_id = self.p_fit.argmax()
            a=self.function(self.gbest)-self.p_fit[g_best_id]
            #print(a)
            ces = self.X[g_best_id]
            #print(ceshi- self.fit)
            print(ces- self.gbest)
            '''

            # 输出当前最优值
            '''
            self.gbest = self.gbest.astype(np.int32)
            print(f"迭代次数为：{iter}",end="‘
             ")
            print(self.gbest, end=" ")
            print(self.fit)
                        if iter > 100 and (fitness[iter - 1] == fitness[iter - 51]) and iter % 40 == 0:
                self.i1 = 0.6
                self.ustks = 1
            '''
            #print(self.fit)
            # 更新V
            # self.V[i] = self.低维重量 * self.stick[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) +
            # self.c2 * self.r2 * (self.gbest - self.X[i])

            # print(self.i1+self.i2+self.i3)
            #print(self.X)
            # print(self.i2 * ( np.equal(self.pbest, self.X)))
            print(iter)
            print(self.gbest)
            print(self.fit)


            for i in range(self.pn):

                #惯性部分
                X_copy = self.X[i].copy()
                for i in range(20):
                    parity = random.random()
                    if parity < 0.5:
                        triangle_1 = random.randint(1, self.dim - 3)
                        triangle_2 = triangle_1+1
                        triangle_3 = triangle_1+2
                        A = self.X[i][triangle_1]
                        B = self.X[i][triangle_2]
                        C = self.X[i][triangle_3]
                        ABC = self.distance_graph[A][B] + self.distance_graph[B][C]
                        ACB = self.distance_graph[A][C] + self.distance_graph[C][B]
                        BAC = self.distance_graph[B][A] + self.distance_graph[A][C]
                        if BAC < ACB < ABC:
                            self.X[i][triangle_2] = A
                            self.X[i][triangle_1] = B
                        if ACB < BAC < ABC:
                            self.X[i][triangle_2] = C
                            self.X[i][triangle_3] = B
                        if self.function(self.X[i]) < self.p_fit[i]:
                            self.p_fit[i] = self.function(self.X[i])
                            self.pbest[i] = self.X[i].copy()
                            #print('yes,pbest更新了______________')
                            if self.function(self.X[i]) < self.fit:
                                self.gbest = self.X[i].copy()
                                self.fit = self.function(self.X[i])
                                #print('yes,gbest更新了+++++++++++++++++')

                    else:
                        triangle_1 = random.randint(1, self.dim - 4)
                        triangle_2 = triangle_1 + 1
                        triangle_3 = triangle_1 + 2
                        triangle_4 = triangle_1 + 3
                        A = self.X[i][triangle_1]
                        B = self.X[i][triangle_2]
                        C = self.X[i][triangle_3]
                        D = self.X[i][triangle_4]
                        ABCD = self.distance_graph[A][B] + self.distance_graph[B][C] + self.distance_graph[C][D]
                        BCDA = self.distance_graph[B][C] + self.distance_graph[C][D] + self.distance_graph[D][A]
                        CDAB = self.distance_graph[C][D] + self.distance_graph[D][A] + self.distance_graph[A][B]
                        DABC = self.distance_graph[D][A] + self.distance_graph[A][B] + self.distance_graph[B][C]

                        if DABC < min(BCDA, CDAB) < ABCD:
                            self.X[i][triangle_1] = D
                            self.X[i][triangle_2] = A
                            self.X[i][triangle_3] = B
                            self.X[i][triangle_4] = C
                        if CDAB < min(DABC, BCDA) < ABCD:
                            self.X[i][triangle_1] = C
                            self.X[i][triangle_2] = D
                            self.X[i][triangle_3] = A
                            self.X[i][triangle_4] = B
                        if BCDA < min(DABC, CDAB) < ABCD:
                            self.X[i][triangle_1] = B
                            self.X[i][triangle_2] = C
                            self.X[i][triangle_3] = D
                            self.X[i][triangle_4] = A
                        if self.function(self.X[i]) < self.p_fit[i]:
                            #print('yes,pbest更新了______________')
                            self.p_fit[i] = self.function(self.X[i])
                            self.pbest[i] = self.X[i].copy()
                            if self.function(self.X[i]) < self.fit:
                                self.gbest = self.X[i].copy()
                                self.fit = self.function(self.X[i])
                                #print('yes,gbest更新了+++++++++++++++++')

                #学习部分
                point_1 = random.randint(1, self.dim - 1)
                #point2 = random.randint(1, self.dim - 1)
                point_2 = point_1+random.randint(2, 8)
                fragment_1 = self.pbest[i][point_1:point_2].copy()
                fragment_2 = self.gbest[point_1:point_2].copy()
                r_ = random.random()
                if r_ > 0.8:
                    # 取gbest片段
                    for j in range(len(fragment_1)):
                        temp = self.X[i][point_1 + j]
                        self.X[i][point_1 + j] = fragment_1[j]
                        self.X[i][np.argwhere(self.X[i] == fragment_1[j])[0]] = temp
                        if self.function(self.X[i]) < self.p_fit[i]:
                            self.p_fit[i] = self.function(self.X[i])
                            self.pbest[i] = self.X[i].copy()
                            #print('yes,pbest更新了______________')
                            if self.function(self.X[i]) < self.fit:
                                self.gbest = self.X[i].copy()
                                self.fit = self.function(self.X[i])
                                print('yes,gbest更新了+++++++++++++++++')
                else:
                    for j in range(len(fragment_2)):
                        temp = self.X[i][point_1 + j]
                        self.X[i][point_1 + j] = fragment_2[j]
                        self.X[i][np.argwhere(self.X[i] == fragment_2[j])[0]] = temp
                        if self.function(self.X[i]) < self.p_fit[i]:
                            self.p_fit[i] = self.function(self.X[i])
                            self.pbest[i] = self.X[i].copy()
                            #print('yes,pbest更新了______________')
                            if self.function(self.X[i]) < self.fit:
                                self.gbest = self.X[i].copy()
                                self.fit = self.function(self.X[i])
                                #print('yes,gbest更新了+++++++++++++++++')
                '''        
                else:
                    for j in range(20):
                        temp = self.X[i][point_1]
                        self.X[i][point_1] = self.gbest[point_1]
                        self.X[i][np.argwhere(self.X[i] == self.gbest[point_1])[0]] = temp
                '''

           #先驱粒子
#########################################
            poineer = self.gbest.copy()
            for i in range (100):
                r = random.random()
                # 任意两交换
                if r < 0.3:
                    ch1 = random.randint(1, self.dim-1)
                    ch2 = random.randint(1, self.dim-1)
                    temp_ch = poineer[ch1]
                    poineer[ch1] = poineer[ch2]
                    poineer[ch2] = temp_ch
                # 任意片段逆序
                if 0.3 <= r < 0.6:
                    re_1 = random.randint(1, self.dim-1)
                    re_2 = random.randint(1, self.dim-1)
                    re1 = min(re_1, re_2)
                    re2 = max(re_1, re_2)
                    temp_re = poineer[re1:re2].copy()
                    poineer[re1:re2] = temp_re[::-1].copy()
                # 贪婪交换
                if r >= 0.6:
                    chgr = random.randint(1, self.dim-2)
                    temp_after = poineer[chgr+1]
                    ch_domain = self.distance_graph[poineer[chgr]]
                    new_after = np.argmin(ch_domain)
                    poineer[chgr+1] = new_after
                    chgr2 = np.argwhere(poineer == new_after)[0]
                    poineer[chgr2] = temp_after

                poineer_f = self.function(poineer)
                if poineer_f < self.fit:
                    #print('yes,gbest更新了+++++++++++++++++')
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

city_num = 52
data = np.fromfile('bolin52', sep=' ')
data = data.reshape(city_num, 3)
data1 = np.delete(data, 0, axis=1)
datax = np.delete(data1, 1, axis=1)
x_local = datax.reshape(city_num)
datay = np.delete(data1, 0, axis=1)
y_local = datay.reshape(city_num)
my_tsp = TSP(x_local, y_local)

my_pso = PSO(pn=100, max_iter=1000)
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






