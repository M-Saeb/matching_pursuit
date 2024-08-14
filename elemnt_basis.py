import numpy as np
import matplotlib.pyplot as plt

# Define the domain
x = np.linspace(0, 2 * np.pi, 1000)

# Basis functions (sine and cosine)
f1 = np.sin(x)
f2 = np.cos(x)

# Plot the basis functions
plt.figure(figsize=(8, 4))
plt.plot(x, f1, label='sin(x)', color='r')
plt.plot(x, f2, label='cos(x)', color='b')

# Set labels, legend, and grid
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True)
plt.legend()
plt.show()
