import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community

def ReLU(x):
    F = np.maximum(0, x)
    return F

def gcn_layer(A_hat, D_hat, X, W):
    D_hat = np.diag(D_hat**-0.5)
    F = ReLU((((D_hat.dot(A_hat)).dot(D_hat)).dot(X)).dot(W))
    return F


# create datas of zachary-kerate-club  
G = nx.karate_club_graph()
pos = nx.spring_layout(G)
node_num = G.number_of_nodes()
# partition
partitions = community.best_partition(G)
# draw
color_list = ['#e46860', '#7fbfe9', '#78b41f', '#b41faa', '#e4d760', '#9660e4', '#e49660', '#c0c0c0', '#804000', '#006b80']
color_index = 0
for part in set(partitions.values()):
    node_list = [node for node in partitions.keys() if partitions[node] == part]
    label_list = {k:k for k in node_list}
    nx.draw_networkx_nodes(G, pos=pos, nodelist=node_list, node_size=2000, node_color=color_list[color_index])
    nx.draw_networkx_labels(G,pos=pos, labels=label_list, font_size=20)
    color_index = color_index + 1
nx.draw_networkx_edges(G, pos=pos, with_labels=True)
plt.show()


# create graph(adjacent matrix) with self-loop
A = np.asarray(nx.to_numpy_matrix(G, nodelist=sorted(list(G.nodes()))))
I = np.eye(node_num)
A_hat = A + I

# creete degree matrix
D_hat = np.sum(A_hat, axis=0)

# create weight matrix randomly 
W_1 = np.random.normal(loc=0, scale=1, size=(node_num, 8))
W_2 = np.random.normal(loc=0, scale=1, size=(W_1.shape[1], 4))
W_3 = np.random.normal(loc=0, scale=1, size=(W_2.shape[1], 2))

# define input
X = I

# propagate
H_1 = gcn_layer(A_hat, D_hat, X,   W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
H_3 = gcn_layer(A_hat, D_hat, H_2, W_3)


# output
output = H_3

# draw
output = output
color_index = 0
for part in set(partitions.values()):
    node_list = [node for node in partitions.keys() if partitions[node] == part]
    label_list = {k:k for k in node_list}
    x = output[node_list, 0]
    y = output[node_list, 1]
    plt.scatter(x, y, c=color_list[color_index], s=1500, alpha=0.4, marker='o')
    color_index = color_index + 1
plt.show()
