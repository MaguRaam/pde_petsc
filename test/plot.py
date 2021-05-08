import matplotlib.pyplot as plt
import numpy as np




data = np.loadtxt("plot.dat")


fig,ax = plt.subplots()
plt.plot(data[:,0], data[:,1],'-b',label = 'u',marker = 'o')
plt.plot(data[:,0], data[:,2],'--r',label = 'uexact',marker = 'o')
leg = ax.legend()
plt.title("solution at t = 5")
plt.xlabel("x")
plt.ylabel("u")
plt.show()
