import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


def create_graphs(n_cycle, n_grid):
    # cycle graph: nodes are arranged in a circle
    Gc = nx.cycle_graph(n_cycle)
    Ac = nx.adjacency_matrix(Gc).toarray()

    # grid graph: nodes are arranged in a gird
    Gg = nx.grid_graph(dim=[n_grid, n_grid])
    Ag = nx.adjacency_matrix(Gg).toarray()

    return Ac, Ag

# Create cycle and grid graphs with 5 and 3 nodes respectively
Ac, Ag = create_graphs(5, 3)

Ac, Ag


def compute_matrices_and_entropy(Ac, Ag, k_max=50):
    # Compute Ac^k and Ag^k for each k, and their entropies
    entropy_c, entropy_g = [], []
    Vc, Wc = Ac.shape
    Vg, Wg = Ag.shape

    for k in range(k_max):
        # Compute Ac^k and Ag^k
        Ack = np.linalg.matrix_power(Ac, k)
        Agk = np.linalg.matrix_power(Ag, k)

        # Compute random VxW matrices Mck and Mgk
        Mck = np.random.rand(Vc, Wc)
        Mgk = np.random.rand(Vg, Wg)

        # Normalize Ack and Agk to obtain πc(k) and πg(k)
        pi_c_k = Ack / Ack.sum(axis=1, keepdims=True)
        pi_g_k = Agk / Agk.sum(axis=1, keepdims=True)

        # Compute entropy of each row in πc(k) and πg(k), and take the average
        entropy_c.append(np.mean([entropy(row) for row in pi_c_k]))
        entropy_g.append(np.mean([entropy(row) for row in pi_g_k]))

    return entropy_c, entropy_g

# Compute entropies for cycle and grid graphs
entropy_c, entropy_g = compute_matrices_and_entropy(Ac, Ag)

entropy_c, entropy_g


def plot_entropy(entropy_c, entropy_g):
    plt.figure(figsize=(10, 6))
    plt.plot(entropy_c, label="Cycle graph")
    plt.plot(entropy_g, label="Grid graph")
    plt.xlabel("k")
    plt.ylabel("Entropy")
    plt.title("Entropy of random walks on cycle and grid graphs")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the entropy values
plot_entropy(entropy_c, entropy_g)
