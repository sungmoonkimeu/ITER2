import numpy as np
from scipy import optimize

def f(x):
     y = abs(x[0] ** 2 + x[1] + 2 + x[2] * 5)
     print(y)
     return y

minimum= optimize.fmin(f,[200,100,0],retall=True)
#print(minimum)