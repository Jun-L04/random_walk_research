from qiskit import QuantumCircuit, QuantumRegister
from qiskit import Aer, transpile, assemble
from qiskit.visualization import plot_histogram
import numpy as np


def quantum_walk_step_cycle(N):
    """
    Returns a Quantum Circuit for one step of a quantum walk on a cycle graph of size N.
    """
    # define quantum registers
    position = QuantumRegister(N, name="pos")  # position register
    coin = QuantumRegister(1, name="coin")  # coin register (2-dimensional)

    # quantum circuit for one step of the quantum walk
    qc = QuantumCircuit(position, coin)

    # coin operation: Hadamard on the coin register
    qc.h(coin)

    # shift operation: Depending on the coin, move left or right on the cycle
    for i in range(N):
        qc.cx(coin[0], position[(i+1) % N])  # move right
        qc.x(coin[0])
        qc.cx(coin[0], position[i])  # move left
        qc.x(coin[0])

    return qc


# example usage:
N_cycle = 5
qc_cycle_step = quantum_walk_step_cycle(N_cycle)
print(qc_cycle_step)


def run_quantum_walk(N, k):
    position = QuantumRegister(N, name="pos")
    coin = QuantumRegister(1, name="coin")
    qc = QuantumCircuit(position, coin)
    
    # initialize walker and coin
    #qc.x(position[0])
    #this gives more interesting results
    qc.h(position)
    
    # Apply quantum walk step k times
    for _ in range(k):
        qc.compose(quantum_walk_step_cycle(N), inplace=True)
    
    # Measure position register
    qc.measure_all()
    
    backend = Aer.get_backend('qasm_simulator')
    result = backend.run(qc).result()
    counts = result.get_counts()
    
    total_shots = sum(counts.values())
    probabilities = {key: value / total_shots for key, value in counts.items()}
    
    print("total shots: ", total_shots)
    for idx, (key, value) in enumerate(probabilities.items(), 1):
        print(f"{idx}: {value:.6f}")
    print(f"Sum of all probabilities: {sum(probabilities.values()):.6f}")

    return probabilities



def compute_entropy(probabilities):
    """
    Compute the Shannon entropy of a probability distribution.
    """
    values = probabilities.values()
    entropy = -sum([p * np.log2(p) for p in values if p > 0])  # Exclude probabilities of 0
    return entropy

# Example usage:
N_cycle = 5
k_steps = 64 #64 max
probabilities = run_quantum_walk(N_cycle, k_steps)
entropy = compute_entropy(probabilities)

#print("Probability distribution:", probabilities)

print("Entropy:", entropy)

plot_histogram(probabilities)