#Plot uexact and u calculated as a function of x


import matplotlib.pyplot as plt
import numpy as np



#load data
data1 = np.loadtxt("plot.dat")


fig,ax = plt.subplots()

#plot u and uexact for beta = 1, 10, 100
plt.scatter(data1[:,0], data1[:,1], label = 'u',marker = '.', color='blue')
plt.plot(data1[:,0], data1[:,2],linestyle="-",label = 'uexact', color='red')


leg = ax.legend()
plt.title("advection diffusion at t = 1000")
plt.xlabel("x")
plt.ylabel("u")
plt.show()