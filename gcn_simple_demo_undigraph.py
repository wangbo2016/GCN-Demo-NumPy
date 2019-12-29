import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

# define graph(adjacent matrix), nodes number is 4
A = np.array(
    [
        [0, 1, 0, 1],
        [1, 0, 1, 1], 
        [0, 1, 0, 1],
        [1, 1, 1, 0]
    ],
    dtype=float
    )
print(A)

# draw graph
G = nx.Graph(A)
nx.draw(G, node_size=2000, node_color='#e46860', font_size=24, width=2, arrowsize=20, with_labels=True)
plt.show()


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
F = A.dot(X)
print(F)