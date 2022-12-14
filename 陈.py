import numpy as np

city_num = 535
data = np.fromfile('bolin52', sep=' ')
data = data.reshape(city_num,3)
data1 = np.delete(data,0,axis=1)
datax = np.delete(data1,1,axis=1)
datax = datax.reshape(city_num)
datay = np.delete(data1,0,axis=1)
datay = datay.reshape(city_num)
print(datay)
#print(list(datay))
