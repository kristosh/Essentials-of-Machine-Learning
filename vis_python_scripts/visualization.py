import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.axis([-5, 5, -5, 5])  # Set axis limits
ax.grid(True)  # Add gridlines for better visualization

# Move left y-axis and bottom x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Label the axes
ax.set_xlabel('X-axis', fontsize=12)
ax.set_ylabel('Y-axis', fontsize=12)

# Set x-axis limits to range between 0 and 4
ax.set_xlim(-2, 4)

# Ensure the y-axis limits remain unchanged
ax.set_ylim(-2, 5)

# Add gridlines for better visualization
ax.grid(True)

vectors = np.array([[1, 1], [1, 2]])  # Define x and y vectors
colors = ['red', 'blue']  # Colors for x and y vectors

# Calculate z as the sum of x and y
z = vectors[0] + vectors[1]  # z = x + y

# Add arrows to the axes
ax.annotate('', xy=(4, 0), xytext=(-2, 0), arrowprops=dict(facecolor='black', shrink=0, width=1, headwidth=8))
ax.annotate('', xy=(0, 5), xytext=(0, -2), arrowprops=dict(facecolor='black', shrink=0, width=1, headwidth=8))

# Plot x and y vectors with different colors
for i in range(len(vectors)):
    # Draw the line representing the vector
    ax.plot([0, vectors[i][0]], [0, vectors[i][1]], color=colors[i], linewidth=2)
    # Draw the arrow at the end of the vector
    ax.arrow(0, 0, vectors[i][0], vectors[i][1], 
             head_width=0.15, head_length=0.15, fc=colors[i], ec=colors[i], length_includes_head=True)
    # Add labels to the vectors
    label = f'x = [1, 1]' if i == 0 else f'y = [1, 2]'
    ax.text(vectors[i][0] + 0.1, vectors[i][1] + 0.1, label, fontsize=12, color=colors[i])

# Plot z vector (resultant of x + y)
ax.plot([0, z[0]], [0, z[1]], color='green', linestyle='dashed', linewidth=2)
ax.arrow(0, 0, z[0], z[1], head_width=0.15, head_length=0.15, fc='green', ec='green', length_includes_head=True)
ax.text(z[0] + 0.1, z[1] + 0.1, 'z = x + y = [2, 3]', fontsize=12, color='green')

plt.show()