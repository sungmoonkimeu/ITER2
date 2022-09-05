from scipy import optimize

def f(x):
    a = input(" ddd: ")
    a = int(a)
    return (a+1)**2

minimum = optimize.fmin(f, 1, maxiter=1)
