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

def opciones_clas(n):
    if n == 1:
        a = {'1': 1}
        x = [a]
    elif n == 2:
        a0 = {'00': 1}
        a1 = {'01': 1}
        a2 = {'10': 1}
        a3 = {'11': 1}
        x = [a0, a1, a2, a3]
    return np.random.choice(x)   

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
                if tipo == 'c':
                    measurement = opciones_clas(jug)
                        
                elif tipo == 'q':
                    measurement = opciones_cuan(jug)
                    """
                    circ = crear_circuito(jug)
                    backend = Aer.get_backend('qasm_simulator')
                    job = execute(circ, backend=backend, shots=1)
                    result = job.result()
                    measurement = result.get_counts(circ)
                    """

                for k,i in enumerate(list(measurement.keys())[0]):
                    if i=='1':
                        ganadores.append(lista[2*j + k])                    
            lista = ganadores   
            m = len(lista)         
    return lista

n1 = 10                                                                                         # cantidad de ciudades
n2_array = np.arange(int(np.ceil(0.1 * n1)), int(np.ceil(5 * n1)), int(np.ceil(0.1 * n1)))      # cantidad de paquetes
n3 = 2                                                                                          # distancia máxima
n4 = 100                                                                                        # cantidad de iteraciones

tiempos_totales = []
costes_totales = []
for tipo in ['c', 'q']:
    if tipo == 'c':
        version = "clásica"
        tests = n4
        print("RESULTADOS DEL JUEGO CLÁSICO:")
    elif tipo == 'q':
        version = "cuántica"
        tests = n4
        print("RESULTADOS DEL JUEGO CUÁNTICO:")
    tiempos = []
    costes = []
    for cant,n2 in enumerate(n2_array):    
        t = 0
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
                t += 1 #hojaldreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
                all_edges = [e for e in net1.edges]
                paquetes_ruta = paquetes_en_ruta(caminitos, all_edges[i])
                #print("Todas las rutas", all_edges)
                #print("Ruta disputada:",all_edges[i])
                #print("Rutas de cada paquete:", caminitos)
                #print("Paquetes que disputan:", paquetes_ruta)
                if paquetes_ruta == []:
                    #t -= 1  #hojaldreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
                    #t += 1  #hojaldreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
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
                print("{:0>3} - Coste final = Tiempo/Envio = {}/{} = {}".format(p+1, tiemp, envio, temp))
            coste += temp   
        t = t / tests
        tiempos.append(t)
        coste = coste / tests
        costes.append(coste)
        print("{:0>3} - Versión {} para {} ciudades y {} paquetes. Coste = {}. Tiempo = {}\n".format(cant+1, version, n1, n2, coste, t))
    tiempos_totales.append(tiempos)
    costes_totales.append(costes)    

"""

plt.figure(figsize=(10,6))
plt.plot(n2_array,costes_totales[0],'blue', label = 'Classical')
plt.plot(n2_array,costes_totales[1],'red', label = 'Quantum')
plt.legend()
plt.title("Cost of classical and quantum protocol \n depending on the number of packages ({} nodes)".format(n1))
plt.show()

"""

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
nx.draw(net2,node_color='red',edge_color = edge_color_list, with_labels = True, width=edge_weights_list)
plt.show() 



fig, axs = plt.subplots(1, 2,figsize=(18,6))

axs[0].set_title("Cost of classical and quantum protocol \n depending on the number of packages ({} nodes)".format(n1))
axs[0].plot(n2_array,costes_totales[0],'blue', label = 'Classical', marker='o')
axs[0].plot(n2_array,costes_totales[1],'red', label = 'Quantum', marker='o')
axs[0].set_xlabel('Number of packages')
axs[0].set_ylabel('Cost')

axs[1].set_title("Number of attempts to connect the source \nto the destination of the packets ({} nodes)".format(n1))
axs[1].plot(n2_array,tiempos_totales[0],'blue', label = 'Classical', marker='o')
axs[1].plot(n2_array,tiempos_totales[1],'red', label = 'Quantum', marker='o')
axs[1].set_xlabel('Number of packages')
axs[1].set_ylabel('Times')

axs[0].legend()
axs[1].legend()
plt.show()