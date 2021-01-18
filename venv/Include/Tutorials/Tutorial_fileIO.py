from numpy import *
DataIN = loadtxt('input.dat')
print(DataIN)
print(DataIN[:,1])
x,y,yerr = loadtxt('input.dat',unpack=True)
print(x, y, yerr)
x,y = loadtxt('input.dat',unpack=True, usecols=[0,1])
print(x, y)


v = arange(1,10,1)
p = 0.15 + v/10.0
savetxt('output.dat',(v,p))
# --> row 1 = v, row2 = p

Dataout = column_stack((v,p))
savetxt('output2.dat',Dataout)
# --> column1 = v, column2 = p

savetxt('output3.dat', Dataout, fmt=('%3i', '%4.3f'))
# --> formatting

DataIN = loadtxt('input.dat')
Dataout = column_stack((DataIN,DataIN[:,1]))
savetxt('output4.dat',Dataout)
