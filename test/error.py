import numpy
from matplotlib import pyplot, cm
from math import log

def convergence_plot(Ngpts,l2,linfty,filename):
  fig,ax = pyplot.subplots()
  pyplot.loglog(N, linfty,'-b',label = 'Linfty error',marker = 'o')
  pyplot.loglog(N, l2,'--r',label = 'L2 error',marker = 'o')
  leg = ax.legend()
  pyplot.title("Convergence in space")
  pyplot.xlabel("N")
  pyplot.ylabel("Error")
  pyplot.savefig(filename)

def convergence_rate(error):
  for i in range(1,numpy.size(error)):
    covergence_rate = (log(error[i-1]) - log(error[i]))/log(2)
    print(covergence_rate)



N  = numpy.array([25,50,100,200,400])
linfty = numpy.array([1.37484e-04, 3.23376e-05, 8.37462e-06, 2.08591e-06, 5.20994e-07])  
l2     = numpy.array([1.00234e-04, 2.38060e-05, 5.89236e-06, 1.47128e-06, 3.67939e-07])
convergence_plot(N,l2,linfty,"error")

#L2 rate of convergence
print("L2 Rate of convergence:")
convergence_rate(l2)

#linfty rate of convergence:
print("Linfty Rate of convergence")
convergence_rate(linfty)