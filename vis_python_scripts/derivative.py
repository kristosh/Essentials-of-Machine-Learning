import numpy as np
import matplotlib.pyplot as plt

# Define function
def f(x):
    return np.exp(0.3 * x)  # example increasing function

# Values
x0 = 2
dx = 1.0
x1 = x0 + dx

y0 = f(x0)
y1 = f(x1)
dy = y1 - y0

# Plot curve
x = np.linspace(0, 6, 400)
plt.plot(x, f(x), 'r', label="$f(x)$", linewidth=2)

# Mark points
plt.plot(x0, y0, 'ko')
plt.plot(x1, y1, 'ko')

# Draw triangle (delta x, delta y)
plt.plot([x0, x1], [y0, y0], 'k--')  # horizontal (dx)
plt.plot([x1, x1], [y0, y1], 'k--')  # vertical (dy)
plt.plot([x0, x1], [y0, y1], 'b')    # secant line

# Annotations
plt.text(x0 - 0.3, y0, "$f(x_0)$", fontsize=12)
plt.text(x1 + 0.1, y1, "$f(x_0 + \delta x)$", fontsize=12)
plt.text((x0 + x1) / 2, y0 - 0.3, r"$\delta x$", fontsize=12)
plt.text(x1 + 0.1, (y0 + y1) / 2, r"$\delta y$", fontsize=12)

# Axes
plt.axhline(0, color='k', linewidth=1)
plt.axvline(0, color='k', linewidth=1)

# Labels
plt.xlabel("x", fontsize=12, fontweight="bold")
plt.ylabel("y", fontsize=12, fontweight="bold")

plt.xticks([])
plt.yticks([])
plt.legend(loc="upper left", fontsize=12)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()