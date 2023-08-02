import datetime
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# have to run in "interactive window"

def create_graphs(n_cycle, n_grid):
    # cycle graph: nodes are arranged in a circle
    Gc = nx.cycle_graph(n_cycle)
    Ac = nx.adjacency_matrix(Gc).toarray()

    # grid graph: nodes are arranged in a gird
    Gg = nx.grid_graph(dim=[n_grid, n_grid])
    Ag = nx.adjacency_matrix(Gg).toarray()

    return Ac, Ag
    # the graphs are not randomly generated
    # but is fine since the walk is random

# create cycle and grid graphs with 5 and 3 nodes respectively
# adjacency matrices 
Ac, Ag = create_graphs(15, 15)

# print them 
print("Cycle graph")
print(Ac)
print("Grid graph")
print(Ag)

# calculate A^k where k is steps of random walk
# the resulting matrix's entries are 
# the number of different walks of 
# length k, between the two nodes

def compute_matrices_and_entropy(Ac, Ag, k_max=1000):
    # compute Ac^k and Ag^k for each k, and their entropies
    entropy_c, entropy_g = [], []
    Vc, Wc = Ac.shape
    Vg, Wg = Ag.shape

    for k in range(k_max):
        # compute Ac^k and Ag^k
        Ack = np.linalg.matrix_power(Ac, k)
        Agk = np.linalg.matrix_power(Ag, k)

        # compute random VxW matrices Mck and Mgk
        # with row/columne dimensions derived from .shape 
        # but why? 
        Mck = np.random.rand(Vc, Wc)
        Mgk = np.random.rand(Vg, Wg)

        # normalize Ac^k and Ag^k to obtain πc(k) and πg(k)
        # normalize by dividing each row with its sum, returns the
        # probability distribution of ending up at each node after k steps
        pi_c_k = Ack / Ack.sum(axis=1, keepdims=True)
        pi_g_k = Agk / Agk.sum(axis=1, keepdims=True)

        # compute entropy of each row in πc(k) and πg(k), also take the average to disregard starting node
        # apply the entropy formula, entropy tells us the uncertainty/randomness of the walk
        entropy_c.append(np.mean([entropy(row) for row in pi_c_k]))
        entropy_g.append(np.mean([entropy(row) for row in pi_g_k]))

    return entropy_c, entropy_g


# higher the entropy the harder it is 
# to perdict the movement of the random walker
# grid graph seems to have a higher entropy than cycle graph

# compute entropies for cycle and grid graphs
entropy_c, entropy_g = compute_matrices_and_entropy(Ac, Ag)

print("Entropy of cycle graph is: ", entropy_c)
print("Entropy of grid graph is: ", entropy_g)

# method to plot everything and compare the entropies 
def plot_entropy(entropy_c, entropy_g):
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    plt.figure(figsize=(10, 6))
    plt.plot(entropy_c, label="Cycle graph")
    plt.plot(entropy_g, label="Grid graph")
    plt.xlabel("k")
    plt.ylabel("Entropy")
    plt.title("Entropy of random walks on cycle and grid graphs")
    plt.suptitle(f"Generated on {timestamp}", fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.show()

# plot the entropy values
plot_entropy(entropy_c, entropy_g)

# plots how entropy changes as number of steps change
# all seem to increase logarithmically? 