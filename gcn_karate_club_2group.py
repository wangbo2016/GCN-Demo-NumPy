import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community

def ReLU(x):
    F = np.maximum(0, x)
    return F

def gcn_layer(A_hat, D_hat, X, W):
    D_hat = np.diag(D_hat**-1)
    F = ReLU(((D_hat.dot(A_hat)).dot(X)).dot(W)) 
    return F


# create datas of zachary-kerate-club  
G = nx.karate_club_graph()
pos = nx.spring_layout(G)
node_num = G.number_of_nodes()
# partition
mrhi_nodelist = [i for i in range(node_num) if G.nodes()[i]['club'] == 'Mr. Hi']
mrhi_labels = {k:k for k in mrhi_nodelist}
mrhi_center = [0]
officer_nodelist = [i for i in range(node_num) if G.nodes()[i]['club'] == 'Officer']
officer_labels = {k:k for k in officer_nodelist}
officer_center = [33]
# draw
nx.draw_networkx_nodes(G, pos=pos, nodelist=mrhi_nodelist,    node_size=2000, node_color='#e46860')
nx.draw_networkx_nodes(G, pos=pos, nodelist=mrhi_center,      node_size=2000, node_color='#b4291f')
nx.draw_networkx_labels(G,pos=pos, labels=mrhi_labels)
nx.draw_networkx_nodes(G, pos=pos, nodelist=officer_nodelist, node_size=2000, node_color='#7fbfe9', labels=officer_labels)
nx.draw_networkx_nodes(G, pos=pos, nodelist=officer_center,   node_size=2000, node_color='#1f78b4')
nx.draw_networkx_labels(G,pos=pos, labels=officer_labels)
nx.draw_networkx_edges(G, pos=pos, with_labels=True)
plt.show()


# create graph(adjacent matrix) with self-loop
A = np.asarray(nx.to_numpy_matrix(G, nodelist=sorted(list(G.nodes()))))
I = np.eye(node_num)
A_hat = A + I

# creete degree matrix
D_hat = np.sum(A_hat, axis=0)

# create weight matrix randomly 
W_1 = np.random.normal(loc=0, scale=1, size=(node_num, 4))
W_2 = np.random.normal(loc=0, scale=1, size=(W_1.shape[1], 2))

# define input
X = I

# propagate
H_1 = gcn_layer(A_hat, D_hat, X,   W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)

# output
output = H_2

# draw
output = output
mrhi_x = output[mrhi_nodelist, 0]
mrhi_y = output[mrhi_nodelist, 1]
plt.scatter(mrhi_x, mrhi_y, c='red', s=1500, alpha=0.4, marker='o')
officer_x = output[officer_nodelist, 0]
officer_y = output[officer_nodelist, 1]
plt.scatter(officer_x, officer_y, c='blue', s=1500, alpha=0.4, marker='o')
plt.show()
