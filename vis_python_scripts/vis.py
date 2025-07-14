import matplotlib.pyplot as plt
import numpy as np

# Define vectors
v1 = np.array([1, 1])
v2 = np.array([1, 2])

# # # Plot setup
# plt.figure()
# plt.axhline(0, color='grey', lw=1)
# plt.axvline(0, color='grey', lw=1)
# plt.grid(True)
# plt.axis('equal')

# # Draw vectors
# origin = np.array([[0, 0], [0, 0]])  # origin points for vectors
# vectors = np.array([v1, v2]).T       # stack vectors column-wise
# plt.quiver(*origin, *vectors, angles='xy', scale_units='xy', scale=1, color=['r', 'b'])

# # Add labels
# plt.text(v1[0], v1[1], 'v1', fontsize=12)
# plt.text(v2[0], v2[1], 'v2', fontsize=12)

# plt.xlim(-2, 4)
# plt.ylim(-2, 4)
# plt.title("Vector Plot")
# plt.show()

# v3 = v1 + v2  # vector addition

# # Plot original vectors and the result
# plt.figure()
# plt.axhline(0, color='grey', lw=1)
# plt.axvline(0, color='grey', lw=1)
# plt.grid(True)
# plt.axis('equal')

# # Draw vectors
# plt.quiver(0, 0, *v1, angles='xy', scale_units='xy', scale=1, color='r', label='v1')
# plt.quiver(*v1, *v2, angles='xy', scale_units='xy', scale=1, color='b', label='v2 from tip of v1')
# plt.quiver(0, 0, *v3, angles='xy', scale_units='xy', scale=1, color='g', label='v1 + v2')

# # Add legend
# plt.legend()
# plt.xlim(-1, 5)
# plt.ylim(-2, 5)
# plt.title("Vector Addition")
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Define vectors
v1 = np.array([1, 1])
v2 = np.array([1, 2])
v3 = v1 + v2  # vector addition

# Set up the plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)

# Hide default spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Bold X and Y axes with arrows
ax.annotate('', xy=(5, 0), xytext=(-1, 0),
            arrowprops=dict(arrowstyle="->", lw=1.8, color='black'))
ax.annotate('', xy=(0, 5), xytext=(0, -2),
            arrowprops=dict(arrowstyle="->", lw=1.8, color='black'))

# Axis labels
ax.text(5.1, -0.3, 'x', fontsize=14, fontweight='bold')
ax.text(-0.4, 5.1, 'y', fontsize=14, fontweight='bold')

# Draw vectors
ax.quiver(0, 0, *v1, angles='xy', scale_units='xy', scale=1, color='r', label='v1')
ax.quiver(*v1, *v2, angles='xy', scale_units='xy', scale=1, color='b', label='v2 from tip of v1')
ax.quiver(0, 0, *v3, angles='xy', scale_units='xy', scale=1, color='g', label='v1 + v2')

# Vector labels
ax.text(v1[0] + 0.1, v1[1]-0.1, 'v1', fontsize=12, color='r')
ax.text(v1[0] + v2[0] + 0.1, v1[1] + v2[1], 'v1 + v2', fontsize=12, color='g')
ax.text(v1[0] + 0.8, v1[1] + v2[1] / 2 + 0.3, 'v2', fontsize=12, color='b')

# Title and legend
ax.set_title("Vector Addition (v1 + v2)", fontsize=14)
ax.legend(loc='upper right')

plt.show()