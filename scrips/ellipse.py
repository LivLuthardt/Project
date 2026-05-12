import numpy as np
import matplotlib.pyplot as plt

def eTiltAngles(x1, x2):
    #Input two points (x1, y1, x1), (x2, y2, z2), get planar tilts alpha and beta
    theta = ellipseAngle(x1, x2)
    a, b = getEllipse(x1, x2)
    alpha, beta = tiltAngles(a, b, theta) 
    #Angle from horizontal plane instead of vertical
    return alpha, beta

def getEllipse(x1, x2, r = 3.5): 
    #Define coordinates as arrays
    x1, x2 = np.array(x1), np.array(x2)
    #Unit vectors for normal vector N and ellipse vector W
    N = np.array([0.0, 0.0, 1.0])
    N = N / np.linalg.norm(N)
    W = x2 - x1
    W = W / np.linalg.norm(W)
    #Determine major/minor axes of ellipse intersection
    b = r                     #Scaling factor is 1/cos(angle below)
    a = r / abs(np.dot(N, W)) #a=r/cos(angle fiber-normal) ->cos(angle)=|N dot W|
    return a, b 

def ellipseAngle(x1, x2):
    #Determines the ellipse angle theta as the angle of vector u projected on the xy plane
    u = np.array(x2) - np.array(x1)
    ux = u[0]
    uy = u[1]
    theta = np.arctan2(uy, ux)
    return theta

def tiltAngles(a, b, theta):
    #Checks if major axis is larger than minor axis
    axes =np.array([a, b])
    a, b = axes.max(), axes.min()
    gamma = np.arccos(b / a)
    #Redefine alpha and beta as angle decompositions, not vector 
    alpha = np.arctan(np.tan(gamma) * np.cos(theta))
    beta = np.arctan(np.tan(gamma) * np.sin(theta))
    return alpha * 180 / np.pi, beta * 180 / np.pi #Convert to degrees

def getEllipseValues(df):
    xtiltAngles, ytiltAngles, xytiltAngles, alist, blist = [], [], [], [], [] #Init empty lists
    first = True
    for r in df.itertuples(index=True):
        x2 = (r[3], r[4], r[2]) #Current fiber point
        if first: 
            x1 = x2
            tilt = (0, 0) #Can't compute tilt from a single point
            theta = 0 #same for axes and angles
            a, b = 0,0
        else:  
            tilt = eTiltAngles(x1, x2) #Pass the past and current points
            theta = ellipseAngle(x1, x2)
            a, b = getEllipse(x1, x2)

        xtiltAngles.append(tilt[0])
        ytiltAngles.append(tilt[1])
        xytiltAngles.append(theta)
        alist.append(a)
        blist.append(b)
        x1 = x2 #Set the current point to the past point
        first = False
    
    return xtiltAngles, ytiltAngles, xytiltAngles, alist, blist

if __name__ == "__main__":
    print(eTiltAngles([0, 0, 0], [1, 0, 1]))