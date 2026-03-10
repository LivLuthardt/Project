import numpy as np 
def eTiltAngles(x1, x2):
    #Input two points (x1, y1, x1), (x2, y2, z2), get planar tilts alpha and beta
    theta = ellipseAngle(x1, x2)
    a, b = getEllipse(x1, x2)
    alpha, beta = tiltAngles(a, b, theta)
    return alpha, beta

def getEllipse(x1, x2): 
    x1, x2 = np.array(x1), np.array(x2)
    N = np.array([0, 0, 1]) #Normal vector of intersection plane
    W = x2 - x1 #Axis vector of cylinder
    W = W / np.linalg.norm(W)
    A = (np.identity(3) - N @ N.T) @ W #Vector of ellipse major axis
    B = np.cross(A, N)  #Vector of ellipse minor axis
    scaling_factor = 7 / np.linalg.norm(B)  #Scale to 7 micrometer cylinder
    A *= scaling_factor
    B *= scaling_factor
    a, b = np.linalg.norm(A), np.linalg.norm(B)
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
    gamma = np.arccos(b/a)
    alpha = gamma * np.cos(theta)
    beta = gamma * np.sin(theta)
    return alpha, beta

if __name__ == "__main__":
    print(eTiltAngles([0, 0, 0], [0, 2, 1]))

