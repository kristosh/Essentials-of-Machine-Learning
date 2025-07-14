import matplotlib.pyplot as plt
import numpy as np

# Vector and scalar
v1 = np.array([1, 1])
scalar = 2
v_scaled = scalar * v1

# Set up the plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, 4.5)
ax.set_ylim(-1, 4.5)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)

# Hide default spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Draw custom X and Y axes with arrows
ax.annotate('', xy=(4.5, 0), xytext=(-1, 0),
            arrowprops=dict(arrowstyle="->", color='black', lw=1.5))
ax.annotate('', xy=(0, 4.5), xytext=(0, -1),
            arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

# Axis labels
ax.text(4.6, -0.3, 'x', fontsize=14, fontweight='bold')
ax.text(-0.4, 4.6, 'y', fontsize=14, fontweight='bold')

# Plot original vector from origin
ax.quiver(0, 0, *v1, color='blue', scale=1, scale_units='xy', angles='xy', label='v1')

# Offset position for scaled vector (to the right)
offset = np.array([0.09, 0])
ax.quiver(*offset, *v_scaled, color='red', scale=1, scale_units='xy', angles='xy', label='2 × v1')

# Annotations
ax.text(v1[0] -0.25, v1[1] + 0.15, 'v1', fontsize=12, color='blue')
ax.text(v_scaled[0] + offset[0] + 0.1, v_scaled[1] + offset[1], '2 × v1', fontsize=12, color='red')

# Title and legend
ax.set_title("Vector scaled", fontsize=14)
ax.legend(loc='upper right')

plt.show()