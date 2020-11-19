import string
import random
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def cal_distance(path):
    dis = 0
    for i in range(len(path) - 1):
        dis += D[path[i]][path[i + 1]]
    return dis

def plot_graph(adjacency_matrix, figure_title=None, print_shortest_path=False, src_node=None, filename=None,
               added_edges=None, pause=False):
    adjacency_matrix = np.array(adjacency_matrix)
    rows, cols = np.where(adjacency_matrix > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    values = [adjacency_matrix[i][j] for i, j in edges]
    weighted_edges = [(e[0], e[1], values[idx]) for idx, e in enumerate(edges)]
    plt.cla()
    fig = plt.figure(1)
    if figure_title is None:
        plt.title("The shortest path for every node to the target")
    else:
        plt.title(figure_title)
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)
    # plot
    labels = nx.get_edge_attributes(G, 'weight')
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos=pos, with_labels=True, font_size=15)  # set with_labels to False if use node labels
    nodes = nx.draw_networkx_nodes(G, pos, node_color="y")
    nodes.set_edgecolor('black')
    nodes = nx.draw_networkx_nodes(G, pos, nodelist=[0, src_node] if src_node else [0], node_color="g")
    nodes.set_edgecolor('black')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, font_size=15)
    if print_shortest_path:
        print("The shortest path (dijkstra) is showed below: ")
        added_edges = []
        for node in range(1, num_nodes):
            shortest_path = nx.dijkstra_path(G, node, 0)  # [1,0]
            print("{}: {}".format("->".join([str(v) for v in shortest_path]),
                                  nx.dijkstra_path_length(G, node, 0)))
            added_edges += list(zip(shortest_path, shortest_path[1:]))
    if added_edges is not None:
        nx.draw_networkx_edges(G, pos, edgelist=added_edges, edge_color='r', width=2)

    if filename is not None:
        plt.savefig(filename)

    if pause:
        plt.pause(0.3)
    else:
        plt.show()

    # return img for video generation
    img = None
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    w, h = fig.canvas.get_width_height()
    img = img.reshape((h, w, 3))
    return img

def epsilon_greedy(s_curr, q, epsilon):
    # find the potential next states(actions) for current state
    potential_next_states = np.where(np.array(D[s_curr]) > 0)[0]
    if random.random() > epsilon:  # greedy
        q_of_next_states = q[s_curr][potential_next_states]
        s_next = potential_next_states[np.argmax(q_of_next_states)]
    else:  # random select
        s_next = random.choice(potential_next_states)
    return s_next

def q_learning(start_state=3, num_epoch=200, gamma=0.8, epsilon=0.05, alpha=0.1, visualize=True, save_video=False):
    print("-" * 20)
    print("Q-learning begins ...")
    if start_state == 0:
        raise Exception("start node(state) can't be target node(state)!")
    imgs = []  # useful for gif/video generation
    len_of_paths = []
    # init all q(s,a)
    q = np.zeros((num_nodes, num_nodes))  # num_states * num_actions
    for i in range(1, num_epoch + 1):
        s_cur = start_state
        path = [s_cur]
        len_of_path = 0
        while True:
            s_next = epsilon_greedy(s_cur, q, epsilon=epsilon)
            # greedy policy
            s_next_next = epsilon_greedy(s_next, q, epsilon=-0.2)  # epsilon<0, greedy policy
            # update q
            reward = -D[s_cur][s_next]
            delta = reward + gamma * q[s_next, s_next_next] - q[s_cur, s_next]
            q[s_cur, s_next] = q[s_cur, s_next] + alpha * delta
            # update current state
            s_cur = s_next
            len_of_path += -reward
            path.append(s_cur)
            if s_cur == 0:
                break
        len_of_paths.append(len_of_path)
        if visualize:
            img = plot_graph(D, print_shortest_path=False, src_node=start_state,
                             added_edges=list(zip(path[:-1], path[1:])), pause=True,
                             figure_title="q-learning\n {}th epoch: {}".format(i, cal_distance(path)))
            imgs.append(img)
    if visualize:
        plt.show()
    if visualize and save_video:
        print("begin to generate gif/mp4 file...")
        imageio.mimsave("q-learning.gif", imgs, fps=5)  # generate video/gif out of list of images
    # print the best path for start state to target state
    strs = "Best path for node {} to node 0: ".format(start_state)
    strs += "->".join([str(i) for i in path])
    print(strs)
    return 0

if __name__ == '__main__':
    # adjacent matrix
    # the target node is 0
    D = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
         [4, 0, 8, 0, 0, 0, 0, 11, 0],
         [0, 8, 0, 7, 0, 4, 0, 0, 3],
         [0, 0, 7, 0, 9, 14, 0, 0, 0],
         [0, 0, 0, 9, 0, 10, 0, 0, 0],
         [0, 0, 4, 14, 10, 0, 3, 0, 0],
         [0, 0, 0, 0, 0, 3, 0, 3, 4],
         [8, 11, 0, 0, 0, 0, 3, 0, 5],
         [0, 0, 3, 0, 0, 0, 4, 5, 0]]

    num_nodes = len(D)
    q_learning(start_state=3, num_epoch=100, gamma=0.8, epsilon=0.05, alpha=0.1, visualize=True, save_video=False)