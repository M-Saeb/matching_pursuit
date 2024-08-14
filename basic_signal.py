import random
import numpy as np
import matplotlib.pyplot as plt

# setup
np.random.seed( random.randint(1, 100) )

# create the signal
grey_x = np.linspace(0, 10, 50)
grey_y = np.sin(grey_x) + np.random.normal(0, 0.5, len(grey_x))

# create the plot windows
y_max = int(grey_y.max()) + 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.grid(True)
ax2.grid(True)
ax1.set_xlim(0, 10)
ax2.set_xlim(0, 10)
ax1.set_ylim(-y_max, y_max)
ax2.set_ylim(-y_max, y_max)

# Plot the signal
ax1.plot(grey_x, grey_y, color='gray', linewidth=2)  # Plot the signal


# add the dots
indices = np.random.choice(len(grey_x), 30, replace=False)
indices.sort()
fig_1_dot_x_values = []
fig_1_dot_y_values = []
fig_2_dot_x_values = []
fig_2_dot_y_values = []
for i in indices:
    fig_1_dot_x_values.append( grey_x[i] )
    fig_1_dot_y_values.append( grey_y[i] )
    fig_2_dot_x_values.append( grey_x[i] )
    fig_2_dot_y_values.append( 0 )
    ax1.scatter(fig_1_dot_x_values, fig_1_dot_y_values, color='black')  # Highlight random points
    ax2.scatter(fig_2_dot_x_values, fig_2_dot_y_values, color='blue')  # Highlight random points
    plt.pause(0.15)

# create the second signal
plt.pause(0.15)
orange_x = np.linspace(0, 10, 100)
orange_y = np.linspace(0, 0, 100)
ax1.plot(orange_x, orange_y, color='orange', linewidth=2)  # Plot the signal

# create the purple signal
purple_x = np.linspace(0, 10, 100)
purple_y = np.linspace(0, 0, 100)
ax2.plot(purple_x, purple_y, color='purple', linewidth=2)  # Plot the signal

plt.show()
