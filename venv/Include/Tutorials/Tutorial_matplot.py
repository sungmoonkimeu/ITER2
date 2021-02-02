from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

x = np.outer(np.linspace(-2, 2, 30), np.ones(30))
y = x.copy().T # transpose
z = np.cos(x ** 2 + y ** 2)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
ax.view_init(elev=15., azim=34)
ax.tick_params(axis='x', which='major', pad=+20)
plt.setp(ax.xaxis.get_majorticklabels(), ha="right", va="bottom")
plt.show()