import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from neural_network import *

def ripple_function(x, y, z, frequency, amplitude):
    r = np.sqrt(x**2 + y**2)
    return amplitude * np.cos(frequency * r) * np.exp(-0.5*r)

# Create a meshgrid for the coordinates
n = 400
x = np.linspace(-5, 5, n)
y = np.linspace(-5, 5, n)
X, Y = np.meshgrid(x, y)
Z = ripple_function(X, Y, 0, frequency=2, amplitude=0.5)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#ax.set_title('3D Surface with Ripples')

# Add a color bar
#fig.colorbar(surf)

# Show the plot
plt.show()

print(Sigmoid(np.array([-0.181, -1.958])))
