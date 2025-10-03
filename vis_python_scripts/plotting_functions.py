import numpy as np
import plotly.graph_objects as go

# Data
x = np.linspace(-6, 6, 400)
y1 = x
y2 = x**2
y3 = x**3
y4 = np.sin(x) * 2  # scaled down for visibility

# Clip y-values for visibility
y1_clip = np.clip(y1, -25, 25)
y2_clip = np.clip(y2, -25, 25)
y3_clip = np.clip(y3, -25, 25)
y4_clip = np.clip(y4, -25, 25)

# Create figure
fig = go.Figure()

# Add functions
fig.add_trace(go.Scatter(x=x, y=y1_clip, mode='lines', name='y = x',
                         line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=x, y=y2_clip, mode='lines', name='y = x^3',
                         line=dict(color='red', width=3, dash='dash')))
fig.add_trace(go.Scatter(x=x, y=y3_clip, mode='lines', name='y = x^3',
                         line=dict(color='green', width=3, dash='dashdot')))
fig.add_trace(go.Scatter(x=x, y=y4_clip, mode='lines', name='y = 5*sin(x)',
                         line=dict(color='purple', width=3, dash='dot')))

# Add arrows for axes
arrow_length = 5
fig.add_annotation(x=arrow_length, y=0, ax=-5, ay=0,
                   showarrow=True, arrowhead=3, arrowsize=2, arrowwidth=2)
fig.add_annotation(x=0, y=25, ax=0, ay=-5,
                   showarrow=True, arrowhead=3, arrowsize=2, arrowwidth=2)

# Update layout: bold labels, title, axes, grid, equal scaling
fig.update_layout(
    title=dict(text="Interactive Plot of Multiple Functions (Equal Scale)", 
               font=dict(size=20, family='Arial', color='black')),
    xaxis=dict(title='x', title_font=dict(size=16, family='Arial', color='black'),
               showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=False,
               range=[-5,5]),
    yaxis=dict(title='y', title_font=dict(size=16, family='Arial', color='black'),
               showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=False,
               range=[-25,25],
               scaleanchor="x"),  # equal scaling
    font=dict(family='Arial', size=12, color='black'),
    legend=dict(title='Functions', font=dict(size=12, family='Arial', color='black')),
    width=700,
    height=700
)

fig.show()