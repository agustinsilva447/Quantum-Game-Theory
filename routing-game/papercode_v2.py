import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator
from qiskit.extensions import RXGate, RYGate, RZGate

def generar_mapa():
    a = 0
    while (np.linalg.matrix_rank(a)!=n1):
        a = np.random.randint(n3, size=(n1,n1))
        np.fill_diagonal(a,0)
        a = np.tril(a) + np.tril(a, -1).T
    return a

def generar_red(a):
    net1 = nx.from_numpy_matrix(copy.deepcopy(a))
    for e in net1.edges():
        net1[e[0]][e[1]]['color'] = 'black'
    edge_weights_list = [net1[e[0]][e[1]]['weight'] for e in net1.edges()]
    return net1, edge_weights_list

def generar_paquetes(n1,n2):
    moves = np.zeros([n2, 2])    
    rng = np.random.default_rng()
    for i in range(n2):
        moves[i,:] = rng.choice(n1, size=2, replace=False)    
    colores = []
    for i in range(n2):
        color = np.base_repr(np.random.choice(16777215), base=16)
        colores.append('#{:0>6}'.format(color))    
    return moves, colores



def caminos(net1, moves):
    caminitos = []
    i = 0
    for j in range(len(moves)):
        cam = []
        try:
            p = nx.dijkstra_path(net1,int(moves[j,0]),int(moves[j,1]))
            for e in range(len(p)-1):
                cam.append(tuple(sorted((p[e], p[e+1]))))    
        except:
            i += 1
            if i == len(moves):
                return caminitos, True        
        caminitos.append(cam)
    return caminitos, False

def paquetes_en_ruta(camin, ruta):
    lista = []
    for i in range(n2):
        if ruta in camin[i]:
            lista.append(i)
    return lista

def opciones_clas(n, tipo):
    if n == 1:
        a = {'1': 1}
        x = [a]
        return np.random.choice(x)
    elif n == 2:
        a0 = {'00': 1}
        a1 = {'01': 1}
        a2 = {'10': 1}
        a3 = {'11': 1}
        x = [a0, a1, a2, a3]
        return np.random.choice(x, p = [tipo*tipo, tipo*(1-tipo), (1-tipo)*tipo, (1-tipo)*(1-tipo)])

def opciones_cuan(n):
    if n == 1:
        a = {'1': 1}
        x = [a]
    elif n == 2:
        a1 = {'01': 1}
        a2 = {'10': 1}
        x = [a1, a2]
    return np.random.choice(x)     

def crear_circuito(n):
    I_f = np.array([[1, 0],
                  [0, 1]]) 
    I = np.array([[1, 0],
                  [0, 1]])
    X_f = np.array([[0, 1],
                  [1, 0]]) 
    X = np.array([[0, 1],
                  [1, 0]])    
    for q in range(n-1):
        I_f = np.kron(I_f, I)
        X_f = np.kron(X_f, X)
    J = Operator(1 / np.sqrt(2) * (I_f + 1j * X_f))    
    J_dg = J.adjoint()
    if n==1:
        dx = np.pi
        dy = 0
        dz = 0
    elif n==2:    
        dx = np.pi/2
        dy = np.pi/4
        dz = 0
    elif n==4:
        dx = np.pi/2
        dy = 3 * np.pi/8
        dz = 3 * np.pi/4
    circ = QuantumCircuit(n,n)
    circ.append(J, range(n))
    for q in range(n):
        circ.append(RXGate(dx),[q])
        circ.append(RYGate(dy),[q])
        circ.append(RZGate(dz),[q])            
    circ.append(J_dg, range(n))
    circ.measure(range(n), range(n))  
    return circ

def juego(lista, tipo):
    m = len(lista)
    if m > 0:
        for r in range(int(np.ceil(np.log2(m)))):
            ganadores = []            
            for j in range(int(np.ceil(m/2))):
                jug = 2 - int(m == j+int(np.ceil(m/2)))
                        
                if tipo == 'q':
                    measurement = opciones_cuan(jug)
                    """
                    # esto es para correr el circuito en el simulador de IBM
                    circ = crear_circuito(jug)
                    backend = Aer.get_backend('qasm_simulator')
                    job = execute(circ, backend=backend, shots=1)
                    result = job.result()
                    measurement = result.get_counts(circ)
                    """
                else:
                    measurement = opciones_clas(jug, tipo)

                for k,i in enumerate(list(measurement.keys())[0]):
                    if i=='1':
                        ganadores.append(lista[2*j + k])                    
            lista = ganadores   
            m = len(lista)         
    return lista

n1 = 10                                                                                         # cantidad de ciudades
#n2_array = np.arange(int(np.ceil(0.25 * n1)), int(np.ceil(10 * n1)), int(np.ceil(0.25 * n1)))  # cantidad de paquetes
n2_array = [200]                                                                                # cantidad de paquetes
n3 = 2                                                                                          # distancia máxima
n4 = 1                                                                                          # cantidad de iteraciones
#p1 = [0, 0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #falta quantum                          # probabilidad de ceder
#p1 = [0, 0.25, 0.5, 0.75, 0.9, 'q']                                                            # probabilidad de ceder
p1 = ['q']

tiempos_totales = []
tiempos_totales1 = []
tiempos_totales2 = []
costes_totales = []

for tipo in p1:
    if tipo == 'q':
        version = "cuántica"
        tests = n4
        print("RESULTADOS DEL JUEGO CUÁNTICO:")
    else:
        version = "clásica"
        tests = n4
        print("RESULTADOS DEL JUEGO CLÁSICO (p = {}):".format(tipo))
    tiempos = []
    tiempos1 = []
    tiempos2 = []
    costes = []
    for cant,n2 in enumerate(n2_array):    
        t = 0
        t1 = 0
        t2 = 0
        coste = 0
        for p in range(tests):
            a = generar_mapa()                            # genero matriz
            net1, edge_weights_list = generar_red(a)      # genero red
            net2, edge_weights_list = generar_red(a)      # genero copia de red
            moves, colores = generar_paquetes(n1,n2)      # genero paquetes
            caminitos, flag = caminos(net1, moves)        # caminos óptimos
            all_edges2 = [e for e in net2.edges]
            veces = np.zeros(len(all_edges2))
            i = 0
            tiemp = 0
            envio = 0
            while not flag:
                t += 1 
                t1 += 1
                all_edges = [e for e in net1.edges]
                paquetes_ruta = paquetes_en_ruta(caminitos, all_edges[i])
                #print("Todas las rutas", all_edges)
                #print("Ruta disputada:",all_edges[i])
                #print("Rutas de cada paquete:", caminitos)
                #print("Paquetes que disputan:", paquetes_ruta)
                if paquetes_ruta == []:
                    t1 -= 1  
                    t2 += 1  
                    i += 1
                else:
                    i = 0
                    ganadores = juego(paquetes_ruta, tipo)
                    #print("Ganadores:",ganadores, "\n")
                    for x in range(len(ganadores)):
                        moves[ganadores[x]] = [-1,-2]
                        for y in caminitos[ganadores[x]]:
                            veces[np.where((np.array(all_edges2) == y).all(axis=1))[0][0]] += 1
                            tiemp += 2 * net2[y[0]][y[1]]['weight'] * veces[np.where((np.array(all_edges2) == y).all(axis=1))[0][0]] - 1
                            net1.remove_edges_from([y])
                            net2[y[0]][y[1]]['color'] = colores[envio]
                        envio += 1
                    caminitos, flag = caminos(net1, moves)
            try:
                temp = tiemp/envio    #tiempo de envío por paquete 
            except ZeroDivisionError:
                temp = 2*n3            
            if ((p+1)%(tests/2) == 0):
                print("{:0>3} - Coste final = Tiempo/Envio = {}/{} = {}".format(p+1, int(tiemp), envio, temp))
            coste += temp   
        t = t / tests
        t1 = t1 / tests
        t2 = t2 / tests
        tiempos.append(t)
        tiempos1.append(t1)
        tiempos2.append(t2)
        coste = coste / tests
        costes.append(coste)
        print("{:0>3} - Versión {} (p = {}) para {} ciudades y {} paquetes. Coste = {}. Tiempo = {}\n".format(cant+1, version, tipo, n1, n2, coste, t))
    tiempos_totales.append(tiempos)
    tiempos_totales1.append(tiempos1)
    tiempos_totales2.append(tiempos2)
    costes_totales.append(costes)    

print(crear_circuito(2))
print("La cantidad de paquetes enviados en el gráfico es {}/{}".format(envio, n2))


for e in net2.edges():                         #sirve para n3 = 2
    if net2[e[0]][e[1]]['color']=='black':
        net2[e[0]][e[1]]['weight'] *= 5
    else:
        net2[e[0]][e[1]]['weight'] *=4
edge_color_list = [net2[e[0]][e[1]]['color'] for e in net2.edges()]
edge_weights_list = [net2[e[0]][e[1]]['weight'] for e in net2.edges()]
#nx.draw_circular(net2,node_color='red',edge_color = edge_color_list, with_labels = True, width=edge_weights_list)
nx.draw(net2,node_size=750,node_color='red',edge_color = edge_color_list, with_labels = True, width=edge_weights_list)
plt.show() 

"""

c = ['b', 'g', 'c', 'm', 'y', 'r']

fig, axs = plt.subplots(1, 2, figsize=(20,10))

axs[0].set_title("Cost vs number of packages ({} nodes)".format(n1))
axs[1].set_title("Number of attempts (games) to connect.")

for x,y in enumerate(p1):
    axs[0].plot(n2_array,costes_totales[x], c[x], label = 'Classical (p = {})'.format(y), marker='.')
    axs[1].plot(n2_array,tiempos_totales1[x], c[x], label = 'Classical (p = {})'.format(y), marker='.')
    
axs[0].set_xlabel('Number of packages')
axs[0].set_ylabel('Cost')
axs[1].set_xlabel('Number of packages')
axs[1].set_ylabel('Times')
axs[0].legend()
axs[1].legend()

plt.show()

"""
"""

c = ['b', 'g', 'c', 'm', 'y', 'r']

fig, axs = plt.subplots(2, 3,figsize=(30,20))

axs[0, 0].set_title("Cost vs number of packages ({} nodes)".format(n1))
axs[0, 1].set_title("(Cost * Total Attemps)")
axs[0, 2].set_title("(Cost * Games Attemps)")
axs[1, 0].set_title("Number of attempts (empty) to connect.")
axs[1, 1].set_title("Total number of attempts to connect")
axs[1, 2].set_title("Number of attempts (games) to connect.")

for x,y in enumerate(p1):
    if y == 'q':
        axs[0, 0].plot(n2_array,costes_totales[x], c[x], label = 'Quantum', marker='.')
        axs[0, 1].plot(n2_array,np.array(costes_totales[x]) * np.array(tiempos_totales[5]), c[x], label = 'Quantum', marker='.')
        axs[0, 2].plot(n2_array,np.array(costes_totales[x]) * np.array(tiempos_totales1[5]), c[x], label = 'Quantum', marker='.')
        axs[1, 0].plot(n2_array,tiempos_totales2[x], c[x], label = 'Quantum', marker='.')
        axs[1, 1].plot(n2_array,tiempos_totales[x], c[x], label = 'Quantum', marker='.')
        axs[1, 2].plot(n2_array,tiempos_totales1[x], c[x], label = 'Quantum', marker='.')
    else:
        axs[0, 0].plot(n2_array,costes_totales[x], c[x], label = 'Classical (p = {})'.format(y), marker='.')    
        axs[0, 1].plot(n2_array,np.array(costes_totales[x]) * np.array(tiempos_totales[x]), c[x], label = 'Classical (p = {})'.format(y), marker='.')
        axs[0, 2].plot(n2_array,np.array(costes_totales[x]) * np.array(tiempos_totales1[x]), c[x], label = 'Classical (p = {})'.format(y), marker='.')
        axs[1, 0].plot(n2_array,tiempos_totales2[x], c[x], label = 'Classical (p = {})'.format(y), marker='.')
        axs[1, 1].plot(n2_array,tiempos_totales[x], c[x], label = 'Classical (p = {})'.format(y), marker='.')
        axs[1, 2].plot(n2_array,tiempos_totales1[x], c[x], label = 'Classical (p = {})'.format(y), marker='.')

axs[1, 0].set_xlabel('Number of packages')
axs[1, 1].set_xlabel('Number of packages')
axs[1, 2].set_xlabel('Number of packages')
axs[0, 0].set_ylabel('Cost')
axs[1, 0].set_ylabel('Times')

axs[0, 0].legend(loc='upper left')
axs[1, 0].legend(loc='upper right')

plt.show()

"""
"""

costs_list = []
times_list = []
plt.title("Trade-off grapf for 20 nodes")
for x,y in enumerate(p1):
    if y == 'q':
        plt.plot(costes_totales[x][-1], tiempos_totales1[x][-1],'r', label = 'Quantum', marker='o')
    else:
        colors = '#{:0>6}'.format(np.base_repr(np.random.choice(16777215), base=16))
        plt.plot(costes_totales[x][-1], tiempos_totales1[x][-1], color = colors, label = 'Classical (p = {})'.format(y), marker='o')
        costs_list.append(costes_totales[x][-1])
        times_list.append(tiempos_totales1[x][-1])
plt.plot(costs_list, times_list, 'b')
plt.xlabel('Cost per package')
plt.ylabel('Connection time')
plt.legend()
plt.show()

"""