import numpy as np
import nashpy as nash

n_max = 40
"""
A = np.array([[-n_max, -5, -5],
              [-10, -10, -10],
              [-20, -20, -20]])
B = np.array([[-n_max, -20, -30],
              [-10, -20, -30],
              [-10, -20, -30]])
"""              
A = np.array([[-n_max, -5],
              [-10, -10]])
B = np.array([[-n_max, -20],
              [-10, -20]])

juego = nash.Game(A, B)
eqs = np.array(list(juego.support_enumeration()))

print("Matriz A:\n", A)
print("Matriz B:\n", B)
print("Equilibrios:\n",eqs)

for i in range(eqs.shape[0]):
    sigma_r = eqs[i][0]
    sigma_c = eqs[i][1]
    sigmas = juego[sigma_r, sigma_c]
    print("Estrategia {}: Sigmas: {}.".format(i,sigmas))