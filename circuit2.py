import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, Gate
from qiskit.quantum_info import Operator
from qiskit.visualization import circuit_drawer

circ = QuantumCircuit(2,2)

J = Gate(name='J', num_qubits=2, params=[])
J_d = Gate(name='J\dagger', num_qubits=2, params=[])

thetaA = Parameter('\\theta_{A}')
phiA = Parameter('\phi_{A}')
lamA = Parameter('\lambda_{A}')

thetaB = Parameter('\\theta_{B}')
phiB = Parameter('\phi_{B}')
lamB = Parameter('\lambda_{B}')

circ = QuantumCircuit(2,2)
circ.append(J,   [0,1])
circ.u(thetaA,phiA,lamA,0)
circ.u(thetaB,phiB,lamB,1)
circ.append(J_d, [0,1])
circ.measure([0,1], [0,1])

circuit_drawer(circ, output='latex', interactive = True, scale = 0.75)