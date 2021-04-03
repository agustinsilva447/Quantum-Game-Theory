import numpy as np
import matplotlib.pyplot as plt

x = np.arange(2, 11, 0.1)
f0 = 1 / x

f = []

plt.figure(figsize=(10, 6))
plt.plot(x, f0,'red', label = 'Quantum')
for p in np.arange(0,1,0.1):
    f.append(np.power(p, x-1) * (1 - p)) # * ((np.power(2,x))/(np.power(2,x) - 1)))
    colors = '#{:0>6}'.format(np.base_repr(np.random.choice(16777215), base=16))
    plt.plot(x, f[-1],'red', color = colors, label = 'p = {}'.format(np.round(p,2)))
    
plt.xlabel("Number of players")
plt.ylabel("Probability")
plt.legend()
plt.title("Probability of a particular player winning the channel just for himslef")
plt.show()