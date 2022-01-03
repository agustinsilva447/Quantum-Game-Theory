import sympy as sp
import numpy as np

a1, a2, b1, b2, c1, c2, d1, d2 = sp.symbols('a1 a2 b1 b2 c1 c2 d1 d2')
a = np.array([a1, a2])
b = np.array([b1, b2])
c = np.array([c1, c2])
d = np.array([d1, d2])

caso1 = np.kron(np.kron(a,b), np.kron(c,d))
caso2 = np.kron(a, np.kron(b, np.kron(c,d)))

if (caso1 == caso2).all:
    print("agus tenía razón")
else:
    print("rachel tenía razón")