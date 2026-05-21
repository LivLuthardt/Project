import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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
phi_deg, theta_deg, alpha_deg = 35, 45, 30
phi, theta, alpha = np.radians(phi_deg), np.radians(theta_deg), np.radians(alpha_deg)
L, r = 4, 0.4

# Local Axes vectors
e1 = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
ref_y = np.array([-np.sin(phi), np.cos(phi), 0])
ref_z = np.cross(e1, ref_y)
e2 = np.cos(alpha)*ref_y + np.sin(alpha)*ref_z
e3 = np.cross(e1, e2)
E = L * e1

# Plot Cylinder Surface (Made more transparent)
z_vals = np.linspace(0, L, 50)
t_vals = np.linspace(0, 2*np.pi, 50)
Z_mesh, T_mesh = np.meshgrid(z_vals, t_vals)
X_cyl = Z_mesh * e1[0] + r * np.cos(T_mesh) * ref_y[0] + r * np.sin(T_mesh) * ref_z[0]
Y_cyl = Z_mesh * e1[1] + r * np.cos(T_mesh) * ref_y[1] + r * np.sin(T_mesh) * ref_z[1]
Z_cyl = Z_mesh * e1[2] + r * np.cos(T_mesh) * ref_y[2] + r * np.sin(T_mesh) * ref_z[2]

ax.plot_surface(X_cyl, Y_cyl, Z_cyl, color='gray', alpha=0.1, edgecolor='none')
ax.plot([0, E[0]], [0, E[1]], [0, E[2]], 'k-.') # Centerline

# --- Intersecting xy-Plane ---
h = E[2] / 2  # Height of intersection
h_offset = 0.05 # Add small offset for better visibility of intersection ring
plane_size = 5
xx, yy = np.meshgrid(np.linspace(-plane_size/2, plane_size, 10), np.linspace(-plane_size/2, plane_size, 10))
zz = np.full_like(xx, h+h_offset)
# Plane color to a lighter teal (#AFEEEE) and reduce alpha
ax.plot_surface(xx, yy, zz, color='#AFEEEE', alpha=0.2, edgecolor='none')

# --- Highlighted Ellipse (Shaded Entirely Purple) ---
t_ell = np.linspace(0, 2*np.pi, 100)
# Solve Z_cyl = h+h_offset for z_val along the cylinder axis
z_ell_params = (h + h_offset - r * np.cos(t_ell) * ref_y[2] - r * np.sin(t_ell) * ref_z[2]) / e1[2]
# Compute 3D coordinates of the intersection ring
x_ell = z_ell_params * e1[0] + r * np.cos(t_ell) * ref_y[0] + r * np.sin(t_ell) * ref_z[0]
y_ell = z_ell_params * e1[1] + r * np.cos(t_ell) * ref_y[1] + r * np.sin(t_ell) * ref_z[1]
z_ell_3d = np.full_like(t_ell, h + h_offset)


# Add the thin ring
ax.plot(x_ell, y_ell, z_ell_3d, color='purple', lw=1.5)

# Plot Global Axes (x1, x2, x3)
for vec, label in zip(np.eye(3)*6, ['x1', 'x2', 'x3']):
    ax.quiver(0, 0, 0, *vec, color='black', arrow_length_ratio=0.05, lw=1.5)
    ax.text(*(vec * 1.1), label, fontsize=16)

# Plot Local Axes (e1, e2, e3)
for vec, label in zip([e1, e2, e3], ['e1', 'e2', 'e3']):
    ax.quiver(*E, *(vec*2), color='black', arrow_length_ratio=0.1, lw=1.5)
    ax.text(*(E + vec*2.3), label, fontsize=16)

# Draw dashed projection lines
P = np.array([E[0], E[1], 0])
ax.plot([0, P[0]], [0, P[1]], [0, P[2]], 'k--', lw=1)
ax.plot([P[0], E[0]], [P[1], E[1]], [P[2], E[2]], 'k--', lw=1)

# --- Angle Arcs and Labels ---
# Phi (Azimuthal)
t_phi = np.linspace(0, phi, 30)
ax.plot(3 * np.cos(t_phi), 3 * np.sin(t_phi), 0, 'k', lw=1.5)
ax.text(3.3 * np.cos(phi/2), 3.3 * np.sin(phi/2), 0, r'$\phi$', fontsize=18)

# Theta (Polar)
t_theta = np.linspace(0, theta, 30)
p_dir = np.array([np.cos(phi), np.sin(phi), 0])
z_dir = np.array([0, 0, 1])
arc_theta = 3.5 * (np.outer(np.sin(t_theta), p_dir) + np.outer(np.cos(t_theta), z_dir))
ax.plot(arc_theta[:,0], arc_theta[:,1], arc_theta[:,2], 'k', lw=1.5)
ax.text(arc_theta[15,0]*1.1, arc_theta[15,1]*1.1, arc_theta[15,2]*1.1, r'$\theta$', fontsize=18)

# Alpha (Spin)
t_alpha = np.linspace(0, alpha, 30)
arc_alpha = E + 1.2 * (np.outer(np.cos(t_alpha), ref_y) + np.outer(np.sin(t_alpha), ref_z))
ax.plot(arc_alpha[:,0], arc_alpha[:,1], arc_alpha[:,2], 'k', lw=1.5)
alpha_label_pos = E + 1.5 * (np.cos(alpha/2)*ref_y + np.sin(alpha/2)*ref_z)
ax.text(alpha_label_pos[0], alpha_label_pos[1], alpha_label_pos[2], r'$\alpha$', fontsize=18)

ax.set_axis_off()
plt.show()