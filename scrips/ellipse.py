import numpy as np 
import scipy as sp
import Geometry3D as geo

#Dataframe of all coordinates

def getEllipse(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    N = np.array([0, 0, 1])
    W = x2 - x1
    W = W / np.linalg.norm(W)
    A = (np.identity(3) - N @ N.T) @ W
    B = np.cross(A, N)
    scaling_factor = 7 / np.linalg.norm(B)
    A *= scaling_factor
    B *= scaling_factor
    a, b = np.linalg.norm(A), np.linalg.norm(B)
    return a, b

def findTiltAngles(coordinates):
    #Return a list of length n-1 with all tilt angles
    angles = []
    for i, x in enumerate(coordinates[1:]):
        x2, x1 = x, coordinates[i]
        theta = Ellipse_Angle(x1, x2)
        a, b = getEllipse(x1, x2)
        alpha, beta = Tilt_Angles(a, b, theta)
        angles.append((alpha, beta))
    return np.array(angles)

def Ellipse_Angle(x1, x2):
    #Determines the ellipse angle theta as the angle of vector u projected on the xy plane
    u = np.array(x2) - np.array(x1)
    ux = u[0]
    uy = u[1]
    theta = np.arctan2(uy, ux)
    return theta

def Tilt_Angles(a, b, theta):
    #Checks if major axis is larger than minor axis
    axes =np.array([a, b])
    a, b = axes.max(), axes.min()
    gamma = np.arccos(b/a)
    alpha = gamma * np.cos(theta)
    beta = gamma * np.sin(theta)
    """
    #Calculates values for alpha and beta, and then returns them 
    tan2 = np.tan(theta)**2
    alpha = np.arccos(np.sqrt((1 + (b ** 2 / a ** 2) * tan2) / (1 + tan2)))
    beta = np.arccos((b / a) * np.sqrt((1 + tan2) / (1 + (b ** 2 / a **2 ) * tan2)))
    """
    return alpha, beta

test_coordinates = [(0, 0, 0), (0, 0, 1), (0, 1, 2)]
print(findTiltAngles(test_coordinates))



