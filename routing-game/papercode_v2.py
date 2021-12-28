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

def opciones_cuan(n, tipo):
    c = tipo[3]             # factor de coherencia -> \rho_{noisy}=c*\rho+(1-c)*\frac{I}{d}
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
        """
        ibmq_16_melbourne: 	{'00': 837, '01': 3076, '10': 3869, '11': 410}
        ibmq_athens: 		{'00': 204, '01': 3940, '10': 3942, '11': 106}
        ibmq_manila: 		{'00': 370, '01': 4161, '10': 3484, '11': 177}
        ibmq_santiago: 		{'00': 156, '01': 3994, '10': 3923, '11': 119}       
        ibmq_lima:          {'00': 4368, '01': 685, '10': 818, '11': 2321} ó {'00': 304, '01': 3575, '10': 4214, '11': 99}
        ibmq_belem:         {'00': 880, '01': 2825, '10': 3922, '11': 565}
        ibmq_quito:         {'00': 398, '01': 4124, '10': 3544, '11': 126}
        """
        if c == "ibmq_16_melbourne":
            return np.random.choice(x, p = [837/8192,3076/8192,3869/8192,410/8192])     # from IBMQ experience
        if c == "ibmq_athens":
            return np.random.choice(x, p = [204/8192,3940/8192,3942/8192,106/8192])     # from IBMQ experience
        if c == "ibmq_manila":
            return np.random.choice(x, p = [370/8192,4161/8192,3484/8192,177/8192])     # from IBMQ experience
        if c == "ibmq_santiago":
            return np.random.choice(x, p = [156/8192,3994/8192,3923/8192,119/8192])     # from IBMQ experience
        if c == "ibmq_lima":
            return np.random.choice(x, p = [4368/8192,685/8192,818/8192,2321/8192])     # from IBMQ experience
        if c == "ibmq_belem":
            return np.random.choice(x, p = [880/8192,2825/8192,3922/8192,565/8192])     # from IBMQ experience
        if c == "ibmq_quito":
            return np.random.choice(x, p = [398/8192,4124/8192,3544/8192,126/8192])     # from IBMQ experience
        else:
            return np.random.choice(x, p = [0.25 * (1 - c), 0.25 * (1 + c), 0.25 * (1 + c), 0.25 * (1 - c)])

def crear_circuito(n, tipo):
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
    circ = QuantumCircuit(n,n)
    circ.append(J, range(n))
    if n==1:
        dx = np.pi
        dy = 0
        dz = 0
    elif n==2:    
        # Pareto, Nash y Mixta
        #dx = np.pi/2
        #dy = np.pi/4
        #dz = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2])

        # barrido
        dx = tipo[0]
        dy = tipo[1]
        dz = tipo[2]

    """
    elif n==4:
        dx = np.pi/2
        dy = 3 * np.pi/8
        dz = 3 * np.pi/4    
    """

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
                if len(tipo) == 1:
                    measurement = opciones_clas(jug, tipo[0])
                if len(tipo) == 4:
                    """
                    # esto es para simular el circuito con ruido en python
                    measurement = opciones_cuan(jug, tipo)
                    """
                    # esto es para correr el circuito en el simulador de IBM
                    circ = crear_circuito(jug, tipo)
                    backend = Aer.get_backend('qasm_simulator')
                    measurement = execute(circ, backend=backend, shots=1).result().get_counts(circ)
                for k,i in enumerate(list(measurement.keys())[0]):
                    if i=='1':
                        ganadores.append(lista[2*j + k])                    
            lista = ganadores   
            m = len(lista)         
    return lista

def checkear_nozero(check):
    circ = crear_circuito(2, check)
    backend = Aer.get_backend('qasm_simulator')
    measurement = execute(circ, backend=backend, shots=1000).result().get_counts(circ)
    return ['00'] != list(measurement.keys())

n1 = 10                                                                                         # cantidad de ciudades
#n2_array = np.arange(int(0.5*np.ceil(n1)), int(np.ceil(10 * n1)), int(0.5*np.ceil(n1)))        # cantidad de paquetes
n2_array = [10 * n1]                                                                            # cantidad de paquetes
n3 = 10                                                                                         # distancia máxima
n4 = 20                                                                                          # cantidad de iteraciones

p1 = []
#p1 = [[0], [0.25], [0.5], [0.75], [0.9]]
#p1 = [[0], [0.25], [0.5], [0.75], [0.9], [np.pi/2, np.pi/4, 0, 1]]
#p1 = [[0.1], [0.3], [0.5], [0.7], [0.9]]
#p1 = [[0.1], [0.3], [0.5], [0.7], [0.9], [np.pi/2, np.pi/4, 0, 1]]

probas = np.arange(0, 1, 0.01)            
for _p in probas:                       # probabilidades de ceder
    p1.append([_p])
#p1.append([np.pi/2, np.pi/4, 0, 1])     # Pareto sí y Nash no, Puro

"""
probas = np.arange(0.3,0.8,0.1)            
for _p in probas:                       # probabilidades de ceder
    p1.append([_p])

deco = np.arange(-1/3,1,(1/9))               
for c in deco:                          # decoherencia de werner
    p1.append([np.pi/2, np.pi/4, 0, c]) 

devices = ["ibmq_16_melbourne", "ibmq_athens", "ibmq_manila", "ibmq_santiago", "ibmq_lima", "ibmq_belem", "ibmq_quito"]
for c in devices:                       # IBM devices
    p1.append([np.pi/2, np.pi/4, 0, c]) 
"""    

angulos = np.arange(0, 2 * np.pi, np.pi/4)                                                      # rotaciones en x,y,z
for _x in angulos:
    for _y in angulos:
        for _z in angulos:
            check = [_x,_y,_z, 1]
            print(_x,_y,_z)
            if checkear_nozero(check):
                p1.append(check)

tiempos_totales = []
tiempos_totales1 = []
tiempos_totales2 = []
costes_totales = []
drop_rate_total = []

for tipo in p1:
    if len(tipo) == 1:
        version = "CLÁSICO"
        version_2 = "p"
        tests = n4
    if len(tipo) == 4:
        version = "CUÁNTICO"
        version_2 = "[Rx, Ry, Rz, c]"
        tests = n4
    print("RESULTADOS DEL JUEGO {} ({} = {}):".format(version, version_2, tipo))
    tiempos = []
    tiempos1 = []
    tiempos2 = []
    costes = []
    drop_rate = []

    for cant,n2 in enumerate(n2_array):    
        t = 0
        t1 = 0
        t2 = 0
        coste = 0
        dr = 0
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
            dr += (envio)/(n2)
        dr = (dr)/(tests)
        t = t / tests
        t1 = t1 / tests
        t2 = t2 / tests
        drop_rate.append(dr)
        tiempos.append(t)
        tiempos1.append(t1)
        tiempos2.append(t2)
        coste = coste / tests
        costes.append(coste)
        print("{:0>3} - Versión {} ({} = {}) para {} ciudades y {} paquetes. Traveling time = {}. Routing Time = {}\n".format(cant+1, version, version_2, tipo, n1, n2, coste, t1))
    drop_rate_total.append(drop_rate)        
    tiempos_totales.append(tiempos)
    tiempos_totales1.append(tiempos1)
    tiempos_totales2.append(tiempos2)
    costes_totales.append(costes)    

"""
print(crear_circuito(2,tipo))
print("La cantidad de paquetes enviados en el gráfico es {}/{}".format(envio, n2))
"""

"""
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

"""
#print(costes_totales)
#print(tiempos_totales1)
# funcion para 5 probabilidades y 1 cuántica
c = ['b', 'g', 'c', 'm', 'y', 'r']
fig, axs = plt.subplots(1, 2)
axs[0].set_title("Traveling time ({} nodes)".format(n1))
axs[1].set_title("Routing time ({} nodes)".format(n1))
for x,y in enumerate(p1):
    if len(y) == 4:
        axs[0].plot(n2_array,costes_totales[x], c[x], label = 'Quantum', marker='.')
        axs[1].plot(n2_array,tiempos_totales1[x], c[x], label = 'Quantum', marker='.')
    else:
        axs[0].plot(n2_array,costes_totales[x], c[x], label = 'Classical (p = {})'.format(y[0]), marker='.')
        axs[1].plot(n2_array,tiempos_totales1[x], c[x], label = 'Classical (p = {})'.format(y[0]), marker='.')
axs[0].set_xlabel('Number of packets')
axs[0].set_ylabel('Time')
axs[0].legend(loc='upper left')
axs[1].set_xlabel('Number of packets')
axs[1].set_ylabel('Time')
axs[1].legend(loc='upper left')
plt.show()
"""

"""
# funcion para 5 probabilidades y 1 cuántica
c = ['b', 'g', 'c', 'm', 'y', 'r']
fig, axs = plt.subplots(3, 1)
axs[0].set_title("Traveling time ({} nodes)".format(n1))
axs[1].set_title("Routing time ({} nodes)".format(n1))
axs[2].set_title("Total time ({} nodes)".format(n1))
for x,y in enumerate(p1):
    if len(y) == 4:
        axs[0].plot(n2_array,costes_totales[x], c[x], label = 'Quantum', marker='.')
        axs[1].plot(n2_array,tiempos_totales1[x], c[x], label = 'Quantum', marker='.')
        axs[2].plot(n2_array,np.add(costes_totales[x],tiempos_totales1[x]), c[x], label = 'Quantum', marker='.')
    else:
        axs[0].plot(n2_array,costes_totales[x], c[x], label = 'Classical (p = {})'.format(y[0]), marker='.')
        axs[1].plot(n2_array,tiempos_totales1[x], c[x], label = 'Classical (p = {})'.format(y[0]), marker='.')
        axs[2].plot(n2_array,np.add(costes_totales[x],tiempos_totales1[x]), c[x], label = 'Classical (p = {})'.format(y[0]), marker='.')
axs[0].set_ylabel('Time')
axs[0].legend(bbox_to_anchor=(1,1))
axs[1].set_ylabel('Time')
axs[2].set_ylabel('Time')
axs[2].set_xlabel('Number of packets')
plt.show()
"""

"""
c = ['b', 'g', 'c', 'm', 'y', 'r']
for x,y in enumerate(p1):
    if len(y) == 4:
        plt.plot(n2_array,np.add(costes_totales[x],tiempos_totales1[x]), c[x], label = 'Quantum', marker='.')
    else:
        plt.plot(n2_array,np.add(costes_totales[x],tiempos_totales1[x]), c[x], label = 'Classical (p = {})'.format(y[0]), marker='.')
plt.title("Total time ({} nodes)".format(n1))
plt.xlabel("Number of packets")
plt.ylabel("Time")        
plt.legend()        
plt.show()
"""

"""
c = ['b', 'g', 'c', 'm', 'y', 'r']
for x,y in enumerate(p1):
    if len(y) == 4:
        plt.plot(n2_array,drop_rate_total[x], c[x], label = 'Quantum', marker='.')
    else:
        plt.plot(n2_array,drop_rate_total[x], c[x], label = 'Classical (p = {})'.format(y[0]), marker='.')
plt.title("Packets Dropout")
plt.xlabel("Number of packets")
plt.ylabel("% delivery")        
plt.legend()        
plt.show()
"""

"""
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(n2_array, costes_totales[0], tiempos_totales1[0], 'b', label = 'p = {}'.format(p1[0]))
ax.plot3D(n2_array, costes_totales[1], tiempos_totales1[1], 'g', label = 'p = {}'.format(p1[1]))
ax.plot3D(n2_array, costes_totales[2], tiempos_totales1[2], 'c', label = 'p = {}'.format(p1[2]))
ax.plot3D(n2_array, costes_totales[3], tiempos_totales1[3], 'm', label = 'p = {}'.format(p1[3]))
ax.plot3D(n2_array, costes_totales[4], tiempos_totales1[4], 'y', label = 'p = {}'.format(p1[4]))
ax.plot3D(n2_array, costes_totales[5], tiempos_totales1[5], 'r', label = 'quantum')
ax.set_xlabel('Packets Number')
ax.set_ylabel('Traveling Time')
ax.set_zlabel('Routing Time')
ax.legend(bbox_to_anchor=(1,1))
plt.show()
"""

"""
# funcion para 5 probabilidades y 1 cuántica
c = ['b', 'g', 'c', 'm', 'y', 'r']
fig, axs = plt.subplots(2, 3,figsize=(30,20))
axs[0, 0].set_title("Traveling time vs number of packages ({} nodes)".format(n1))
axs[0, 1].set_title("(Traveling time * Total Attemps)")
axs[0, 2].set_title("(Traveling time * Games Attemps)")
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
axs[0, 0].set_ylabel('Traveling time')
axs[1, 0].set_ylabel('Times')
axs[0, 0].legend(loc='upper left')
axs[1, 0].legend(loc='upper right')
plt.show()
"""

max_x = 0
max_y = 0
costs_list_c = []
times_list_c = []
costs_list_q = []
times_list_q = []
plt.title("Trade-off graph for {} nodes".format(n1))
#p1.reverse()
for x,y in enumerate(p1): 
    colors = '#{:0>6}'.format(np.base_repr(np.random.choice(16777215), base=16))
    if len(y) == 1:
        #plt.plot(costes_totales[x][-1], tiempos_totales1[x][-1], color = colors, label = 'Classical (p = {})'.format(np.round(y[0],3)), marker='o')
        plt.plot(costes_totales[x][-1], tiempos_totales1[x][-1],'b', label = 'Classical', marker='o')
        costs_list_c.append(costes_totales[x][-1])
        times_list_c.append(tiempos_totales1[x][-1])
        if costes_totales[x][-1] > max_x:
            max_x = costes_totales[x][-1]
        if tiempos_totales1[x][-1] > max_y:
            max_y = tiempos_totales1[x][-1] 
    if len(y) == 4:
        if type(y[3]) == str:
            plt.plot(costes_totales[x][-1], tiempos_totales1[x][-1], color = colors, label = 'IBMQ = {}'.format(y[3]), marker='o')
        else:
            #plt.plot(costes_totales[x][-1], tiempos_totales1[x][-1], color = colors, label = 'Quantum (c = {})'.format(np.round(y[3],3)), marker='.')
            plt.plot(costes_totales[x][-1], tiempos_totales1[x][-1],'r', label = 'Quantum', marker='.')
            costs_list_q.append(costes_totales[x][-1])
            times_list_q.append(tiempos_totales1[x][-1])  
plt.plot(costs_list_c, times_list_c, 'b') #, label = 'Classical')
#plt.plot(costs_list_q, times_list_q, 'r') #, label = 'Quantum')
plt.xlabel('Traveling time')
plt.ylabel('Routing time')
#plt.xlim(right = 1.1 * max_x)
#plt.ylim(top = 1.1 * max_y)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), ncol=1)
plt.show()