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

    def update(self):

        fitness = []
        T = 1000

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
            T = T*0.98


            for i in range(self.pn):

                X_copy = self.X[i].copy()
                r = random.random()
                # 任意两交换
                if r < 0.2:
                    for j in range(1):
                        ch1 = random.randint(1, self.dim - 1)
                        ch2 = random.randint(1, self.dim - 1)
                        temp_ch = X_copy[ch1]
                        X_copy[ch1] = X_copy[ch2]
                        X_copy[ch2] = temp_ch
                        temp_x_fit = self.function(X_copy)
                        if temp_x_fit < self.x_fit[i]:
                            self.x_fit[i] = temp_x_fit
                            self.X[i] = X_copy.copy()

                        else:
                            r = random.random()
                            det = temp_x_fit - self.x_fit[i]
                            p = math.exp(-det / T)
                            if p > r:
                                self.x_fit[i] = temp_x_fit
                                self.X[i] = X_copy.copy()
                        '''
                        temp_x_fit = self.function(X_copy)                        
                        if temp_x_fit < self.x_fit[i]:
                            self.x_fit[i] = temp_x_fit
                            self.X[i] = X_copy.copy()
                        
                        else:
                            r = random.random()
                            det = temp_x_fit - self.x_fit[i]
                            p = math.exp(-det / T)
                            if p > r:
                                self.x_fit[i] = temp_x_fit
                                self.X[i] = X_copy.copy()
                        '''
                # 任意片段逆序
                if 0.2 <= r < 0.7:
                    re1 = random.randint(1, self.dim - 1)
                    re2 = re1 + random.randint(2, 18)
                    temp_re = X_copy[re1:re2].copy()
                    X_copy[re1:re2] = temp_re[::-1].copy()
                    temp_x_fit = self.function(X_copy)
                    if temp_x_fit < self.x_fit[i]:
                        self.x_fit[i] = temp_x_fit
                        self.X[i] = X_copy.copy()
                    else:
                        r = random.random()
                        det = temp_x_fit - self.x_fit[i]
                        p = math.exp(-det / T)
                        if p > r:
                            self.x_fit[i] = temp_x_fit
                            self.X[i] = X_copy.copy()
                # 贪婪交换
                if r >= 0.7:
                    for i in range(5):
                        chgr = random.randint(1, self.dim - 2)
                        temp_after = X_copy[chgr + 1]
                        ch_domain = self.distance_graph[X_copy[chgr]]
                        new_after = np.argmin(ch_domain)
                        X_copy[chgr + 1] = new_after
                        chgr2 = np.argwhere(X_copy == new_after)[0]
                        X_copy[chgr2] = temp_after
                        temp_x_fit = self.function(X_copy)
                        if temp_x_fit < self.x_fit[i]:
                            self.x_fit[i] = temp_x_fit
                            self.X[i] = X_copy.copy()
                        else:
                            r = random.random()
                            det = temp_x_fit - self.x_fit[i]
                            p = math.exp(-det / T)
                            if p > r:
                                self.x_fit[i] = temp_x_fit
                                self.X[i] = X_copy.copy()



                # 学习部分
                for j in range(self.dim-iter % 2-1):
                    index_before = 0.2*np.argwhere(self.pbest[i] ==self.X[i][j]) + 0.8*np.argwhere(self.gbest == self.X[i][j])
                    index_behind = 0.2*np.argwhere(self.pbest[i] == self.X[i][j+iter % 2+1]) + 0.8*np.argwhere(self.gbest == self.X[i][j+iter % 2+1])
                    if index_behind < index_before:
                        temp = self.X[i][j]
                        self.X[i][j] = self.X[i][j+iter % 2+1]
                        self.X[i][j+iter % 2+1] = temp
                    temp_x_fit = self.function(X_copy)
                    if temp_x_fit < self.x_fit[i]:
                        self.x_fit[i] = temp_x_fit
                        self.X[i] = X_copy.copy()

            '''
            for i in range(self.pn):
                for j in range(self.dim):
                    index_before = 0.3 * np.argwhere(self.pbest[i] == self.X[i][j]) + 0.7 * np.argwhere(
                        self.gbest == self.X[i][j])

                    temp = self.X[i][j]
                    self.X[i][j] = self.X[i][int(index_before[0])]
                    self.X[i][int(index_before[0])] = temp
            '''






           #先驱粒子
#########################################
            poineer = self.gbest.copy()
            for i in range(100):
                r = random.random()
                # 任意两交换
                if r < 0.2:
                    ch1 = random.randint(1, self.dim-1)
                    ch2 = random.randint(1, self.dim-1)
                    temp_ch = poineer[ch1]
                    poineer[ch1] = poineer[ch2]
                    poineer[ch2] = temp_ch
                # 任意片段逆序
                if 0.2 <= r < 0.7:
                    re1 = random.randint(1, self.dim-1)
                    re2 = re1 + random.randint(2, 18)
                    temp_re = poineer[re1:re2].copy()
                    poineer[re1:re2] = temp_re[::-1].copy()
                # 贪婪交换
                if r >= 0.7:
                    chgr = random.randint(1, self.dim-2)
                    temp_after = poineer[chgr+1]
                    ch_domain = self.distance_graph[poineer[chgr]]
                    new_after = np.argmin(ch_domain)
                    poineer[chgr+1] = new_after
                    chgr2 = np.argwhere(poineer == new_after)[0]
                    poineer[chgr2] = temp_after

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






