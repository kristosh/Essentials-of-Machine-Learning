import matplotlib.pyplot as plt
import numpy as np

# Define vectors
v2 = np.array([1, 2])
r = np.array([-2, 1])

# Plot setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-4, 4)
ax.set_ylim(-1, 4)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)

# Remove default spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Draw custom X and Y axes with arrows
ax.annotate('', xy=(4, 0), xytext=(-4, 0),
            arrowprops=dict(arrowstyle="->", lw=1.8, color='black'))
ax.annotate('', xy=(0, 4), xytext=(0, -1),
            arrowprops=dict(arrowstyle="->", lw=1.8, color='black'))

# Axis labels
ax.text(4.1, -0.3, 'x', fontsize=14, fontweight='bold')
ax.text(-0.4, 3.6, 'y', fontsize=14, fontweight='bold')

# Draw vectors from origin
origin = np.array([[0, 0], [0, 0]])  # origin points for vectors
vectors = np.array([v2, r]).T       # stack vectors column-wise
ax.quiver(*origin, *vectors, angles='xy', scale_units='xy', scale=1, color=['r', 'b'])

# Add labels
ax.text(v2[0] + 0.1, v2[1], 'v2', fontsize=12, color='r')
ax.text(r[0] + 0.15, r[1]+0.15, 'r', fontsize=12, color='b')

# Title
ax.set_title("Vector Plot", fontsize=14)

plt.show()