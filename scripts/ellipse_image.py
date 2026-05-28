import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

# Configure Matplotlib for default font rendering
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.size": 16
})

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=55)

# Angles, length, and radius
phi_deg, theta_deg = 35, 60
phi, theta = np.radians(phi_deg), np.radians(theta_deg)
L, r = 5, 0.4

# Local Axes vectors needed for cylinder generation
e1 = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
ref_y = np.array([-np.sin(phi), np.cos(phi), 0])
ref_z = np.cross(e1, ref_y)
E = L * e1

# Plot Cylinder Surface
z_vals = np.linspace(0, L, 50)
t_vals = np.linspace(0, 2*np.pi, 50)
Z_mesh, T_mesh = np.meshgrid(z_vals, t_vals)
X_cyl = Z_mesh * e1[0] + r * np.cos(T_mesh) * ref_y[0] + r * np.sin(T_mesh) * ref_z[0]
Y_cyl = Z_mesh * e1[1] + r * np.cos(T_mesh) * ref_y[1] + r * np.sin(T_mesh) * ref_z[1]
Z_cyl = Z_mesh * e1[2] + r * np.cos(T_mesh) * ref_y[2] + r * np.sin(T_mesh) * ref_z[2]

ax.plot_surface(X_cyl, Y_cyl, Z_cyl, color='gray', alpha=0.1, edgecolor='none')

# Fiber Centerline 
ax.plot([0, E[0]], [0, E[1]], [0, E[2]], color="#B36599", lw=3) 

# --- Intersecting xy-Plane ---
h = 2
h_offset = 0.05 
plane_size = 5
xx, yy = np.meshgrid(np.linspace(-plane_size/2, plane_size, 10), np.linspace(-plane_size/2, plane_size, 10))
zz = np.full_like(xx, h+h_offset)
ax.plot_surface(xx, yy, zz, color="#6392C1", alpha=0.2, edgecolor='none')

# Ellipse
t_ell = np.linspace(0, 2*np.pi, 100)
z_ell_params = (h + h_offset - r * np.cos(t_ell) * ref_y[2] - r * np.sin(t_ell) * ref_z[2]) / e1[2]
x_ell = z_ell_params * e1[0] + r * np.cos(t_ell) * ref_y[0] + r * np.sin(t_ell) * ref_z[0]
y_ell = z_ell_params * e1[1] + r * np.cos(t_ell) * ref_y[1] + r * np.sin(t_ell) * ref_z[1]
z_ell_3d = np.full_like(t_ell, h + h_offset)

ax.plot(x_ell, y_ell, z_ell_3d, color="#6392C1", lw=4, zorder=5)

# Plot Global Axes (x, y, z)
for vec, label in zip(np.eye(3)*6, ['x', 'y', 'z']):
    ax.quiver(0, 0, 0, *vec, color='black', arrow_length_ratio=0.05, lw=1.4)
    ax.text(*(vec * 1.1), label, fontsize=16)

# Normal Vector N 
ax.quiver(0, 0, 0, 0, 0, 4.5, color="#D49E5C", arrow_length_ratio=0.05, lw=2.5)
ax.text(0, 0, 4.8, 'N', color='#D49E5C', fontsize=18, fontweight='bold')

# Projection on xy-plane
P = np.array([E[0], E[1], 0])
ax.plot([0, P[0]], [0, P[1]], [0, P[2]], color='gray', linestyle='--', lw=2)
ax.plot([P[0], E[0]], [P[1], E[1]], [P[2], E[2]], 'k:', lw=1.5)

# Projections on xz and yz planes for context
ax.plot([0, E[0]*1.2], [0, 0], [0, E[2]*1.2], color='gray', linestyle=':', lw=2)
ax.plot([0, 0], [0, E[1]*1.2], [0, E[2]*1.2], color='gray', linestyle=':', lw=2)

# --- Angle Arcs and Labels ---
arc_color = "#6C834F"

# 1. Phi (Azimuthal, in xy-plane)
t_phi = np.linspace(0, phi, 30)
ax.plot(1.8 * np.cos(t_phi), 1.8 * np.sin(t_phi), 0, color=arc_color, lw=2)
ax.text(2.1 * np.cos(phi/2), 2.1 * np.sin(phi/2), 0, r'$\phi$', color=arc_color, fontsize=18)

# 2. Gamma (Largest radius to sit at the top)
t_gamma = np.linspace(0, theta, 30)
p_dir = np.array([np.cos(phi), np.sin(phi), 0])
z_dir = np.array([0, 0, 1])
arc_gamma = 3.5 * (np.outer(np.sin(t_gamma), p_dir) + np.outer(np.cos(t_gamma), z_dir))
ax.plot(arc_gamma[:,0], arc_gamma[:,1], arc_gamma[:,2], color=arc_color, lw=2)
ax.text(arc_gamma[15,0]*1.1, arc_gamma[15,1]*1.1, arc_gamma[15,2]*1.1, r'$\gamma$', color=arc_color, fontsize=18)

# 3. Theta_x (Angle from Z-axis to the projection on XZ plane)
theta_x_max = np.arctan2(e1[0], e1[2]) 
t_tx = np.linspace(0, theta_x_max, 30)
arc_tx = 2.0 * (np.outer(np.sin(t_tx), np.array([1, 0, 0])) + np.outer(np.cos(t_tx), np.array([0, 0, 1])))
ax.plot(arc_tx[:,0], arc_tx[:,1], arc_tx[:,2], color=arc_color, lw=2)
ax.text(arc_tx[15,0]*1.2, 0, arc_tx[15,2]*1.2, r'$\theta_x$', color=arc_color, fontsize=18)

# 4. Theta_y (Angle from Z-axis to the projection on YZ plane)
theta_y_max = np.arctan2(e1[1], e1[2]) 
t_ty = np.linspace(0, theta_y_max, 30)
arc_ty = 2.8 * (np.outer(np.sin(t_ty), np.array([0, 1, 0])) + np.outer(np.cos(t_ty), np.array([0, 0, 1])))
ax.plot(arc_ty[:,0], arc_ty[:,1], arc_ty[:,2], color=arc_color, lw=2)
ax.text(0, arc_ty[15,1]*1.2, arc_ty[15,2]*1.2, r'$\theta_y$', color=arc_color, fontsize=18)

ax.set_axis_off()
plt.show()