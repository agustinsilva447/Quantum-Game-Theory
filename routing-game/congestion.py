import numpy as np 
import matplotlib.pyplot as plt 
from scipy import signal

x = np.arange(0,3,0.01)
y1 = 0.666 * x
y1[int(len(y1)/2)::] = 1

b, a = signal.butter(1, 0.015)
y2 = signal.filtfilt(b, a, y1)
y2[:int(len(y1)/4):] = 1 * y1[:int(len(y1)/4):]

y3 = y2[::-1] * y2

n = 2

plt.plot(x,y3, 'b', linewidth = n, label = 'congestión')
plt.plot(x,y2, 'g', linewidth = n, label = 'deseado')
plt.plot(x,y1, 'r', linewidth = n, label = 'ideal')

plt.hlines(1, 0, 1.5, linewidth = n, colors = 'red', linestyles = 'dashed', label = 'capacidad total')
plt.hlines(0.6, 0, 0.9, linewidth = n, label = 'sin congestión')
plt.vlines(0.9, 0, 0.6, linewidth = n)

plt.xlabel('Paquetes enviados')
plt.ylabel('Paquetes entregados')
plt.xlim(left = 0, right = 2)
plt.ylim(bottom = 0)
plt.legend(loc = 'lower right')
plt.show()