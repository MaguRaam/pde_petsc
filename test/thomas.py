#Plot uexact and u calculated as a functionof x for different values of beta


import matplotlib.pyplot as plt
import numpy as np



#load data for beta = 1, 10, 100
data1 = np.loadtxt("Output1.dat")
data2 = np.loadtxt("Output2.dat")
data3 = np.loadtxt("Output3.dat")


fig,ax = plt.subplots()

#plot u and uexact for beta = 1, 10, 100
plt.plot(data1[:,0], data1[:,1],linestyle="-",label = 'u beta = 1',marker = 'p')
plt.plot(data1[:,0], data1[:,2],linestyle="--",label = 'uexact beta = 1',marker = '*')

plt.plot(data2[:,0], data2[:,1],linestyle="-",label = 'u beta = 10',marker = 'o')
plt.plot(data2[:,0], data2[:,2],linestyle="--",label = 'uexact beta = 10',marker = '^')

plt.plot(data3[:,0], data3[:,1],linestyle="-",label = 'u beta = 100',marker = '.')
plt.plot(data3[:,0], data3[:,2],linestyle="--",label = 'uexact beta = 100',marker = 'x')



leg = ax.legend()
plt.title("u and uexact for beta = 1, 10, 100")
plt.xlabel("x")
plt.ylabel("u")
plt.savefig("results/thomas2")