'''
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot([1, 2, 3, 4])
plt.axis([0, 3, 0, 5])
plt.ylabel('some numbers')
plt.xlabel('x(t)')
plt.title('TEST!!')

plt.figure(2)
plt.subplot(211)
plt.plot([1,2,3])
plt.subplot(212)
plt.plot([4,5,6])
plt.show()

plt.figure(3)
plt.plot([1,5, 10])
plt.text(1,3,r'$\mu=100,\ \sigma = 15$')
plt.grid(True)
'''
# importing matplotlib module and respective classes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator, MaxNLocator)

x = [3, 2, 7, 4, 9]
y = [10, 4, 7, 1, 2]
plt.figure(3)
ax = plt.axes()

ax.set_title('Example Graph')

ax.set_ylabel('y-AXIS')
ax.set_xlabel('x-AXIS')

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Make x-axis with major ticks that
# are multiples of 11 and Label major
# ticks with '% 1.2f' formatting
#ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_locator(MaxNLocator(20))
ax.xaxis.set_major_formatter(FormatStrFormatter('% 1.2f'))

# make x-axis with minor ticks that
# are multiples of 1 and label minor
# ticks with '% 1.2f' formatting
ax.xaxis.set_minor_locator(MultipleLocator(3))
ax.xaxis.set_minor_formatter(FormatStrFormatter('% 1.2f'))

ax.plot(x, y)
plt.show()