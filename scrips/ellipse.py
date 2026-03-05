import numpy as np 
import scipy as sp
import Geometry3D as geo

def getEllipse(x1, x2):
    #takes inputs of the two points (x, y, z) as ndarrays to construct ellipse projection
    #returns ellipse area and length
    p1, p2 = geo.Point(x1), geo.Point(x2)
    v1 = geo.Vector(p1, p2)
    cyl = geo.Cylinder(p1,7,v1,n=100)
    midpoint = geo.Point((x1 + x2)/2)
    plane = geo.Plane(midpoint, geo.z_unit_vector()) # construct at midpoint of cylinder to avoid endpoint cut issues
    ellipse = geo.intersection(cyl, plane)
    return ellipse.length(), ellipse.area()


def findTiltAngles(coordinates):
    #Return a list of length n-1 with all tilt angles
    for i, x in enumerate(coordinates):
        if i == 0: continue
        x2, x1 = x, coordinates[i-1]
        len, area = getEllipse(x1, x2)





initial_guess = [5,5]
solution = sp.optimize.fsolve(system, initial_guess)
print(solution)
def find_semi_axes(Area, Circumference):
    def system(vars):
        a,b = vars
        if a <= 0 or b <= 0:
            return [1e6, 1e6]
        h = (a-b)**2/(a+b)**2
        f_area = Area - np.pi*a*b
        f_circum = Circumference - np.pi*(a+b)*(1+(3*h)/(10+np.sqrt(4-3*h)))
        return [f_area, f_circum]
    initial_guess = [5,5]
    solution = sp.optimize.fsolve(system, initial_guess)
    a,b = solution
    return solution

print(find_semi_axes(37.69911184,26.72978556))
