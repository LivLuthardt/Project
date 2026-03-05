import numpy as np 
import scipy as sp
import Geometry3D as geo

def getEllipse(x1, x2):
    #takes inputs of the two points (x, y, z) as ndarrays to construct ellipse projection
    #returns ellipse area and length
    x3 = x2-x1
    cyl = geo.Cylinder(geo.Point(x1.tolist()),7,x3.tolist(),n=100)
    midpoint = (x1 + x2)/2 
    plane = geo.Plane(midpoint.tolist(), geo.z_axis()) # construct at midpoint of cylinder to avoid endpoint cut issues
    ellipse = geo.intersection(cyl, plane)
    return ellipse.length(), ellipse.area()


def findTiltAngles(coordinates):
    #Return a list of length n-1 with all tilt angles
    for i, x in enumerate(coordinates):
        if i == 0: continue
        x2, x1 = x, coordinates[i-1]
        len, area = getEllipse(x1, x2)





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
