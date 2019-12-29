import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

# define graph(adjacent matrix), nodes number is 4
A = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, 1, 1], 
        [0, 1, 0, 0],
        [1, 0, 1, 0]
    ],
    dtype=float
    )
print(A)

# add self-loop
I = np.array(np.eye(A.shape[0]))
A_hat = A + I
print(A_hat)

# draw graph
G = nx.DiGraph(A_hat)
nx.draw(G, node_size=2000, node_color='#e46860', font_size=24, width=2, arrowsize=20, with_labels=True)
plt.show()



# compute degree matrix
D_hat = np.sum(A_hat, axis=0)
print(D_hat)

# define weight matrix
W = np.array(
    [
        [1, -1],
        [-1, 1]
    ]
    )
print(W)

# define input, format is [4,2]
X = np.array(
    [
        [i, -i]
        for i in range(A.shape[0])
    ], 
    dtype=float
    )
print(X)


# propagate
D_hat = np.diag(D_hat**-0.5)
Z = (((D_hat.dot(A_hat)).dot(D_hat)).dot(X)).dot(W)
print(Z)

# activate
F = relu(Z)
print(F)

