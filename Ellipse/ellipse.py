import numpy as np 
import scipy as sp









Area = 37.69911184
Circum = 26.72978556


def system(vars):
    a,b = vars
    if a <= 0 or b <= 0:
        return [1e6, 1e6]
    h = (a-b)**2/(a+b)**2
    f_area = Area - np.pi*a*b
    f_circum = Circum - np.pi*(a+b)*(1+(3*h)/(10+np.sqrt(4-3*h)))
    return [f_area, f_circum]

initial_guess = [5,5]
solution = sp.optimize.fsolve(system, initial_guess)
print(solution)