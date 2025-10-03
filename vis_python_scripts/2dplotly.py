import numpy as np
import plotly.graph_objects as go

# Define grid
a= np.linspace(-5, 5, 100)
b = np.linspace(-5, 5, 100)
A, B = np.meshgrid(a, b)
Y = A**2 + B**2  # paraboloid

# Create 3D surface plot
fig = go.Figure(data=[go.Surface(y=Y, a=A, b=B, colorscale="Viridis")])

# Update layout
fig.update_layout(
    title="Paraboloid: y = x_1^{2} + x_2^{2}",
    scene=dict(
        xaxis_title="x_1",
        yaxis_title="x_2",
        zaxis_title="y"
    ),
    width=700,
    height=700
)

fig.show()