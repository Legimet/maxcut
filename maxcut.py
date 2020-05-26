import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt

class MaxCut:
    def __init__(self, graph):
        self.qubits = len(graph)
        self.hc = np.zeros(1 << self.qubits)
        self.graph = graph.copy()
        if not nx.get_edge_attributes(self.graph, 'weight'):
            for (u, v) in self.graph.edges:
                self.graph[u][v]['weight'] = 1
        self.total_weight = sum([e[2] for e in self.graph.edges(data='weight')])
        for i in range(self.hc.size):
            bitstr = np.binary_repr(i, width=self.qubits)
            for (u, v, w) in self.graph.edges(data='weight'):
                if bitstr[u] != bitstr[v]:
                    self.hc[i] += w

    def random_edges(self, n=1):
        edges_wt = list(self.graph.edges(data='weight'))
        edges = np.array([e[:2] for e in edges_wt])
        probs = np.array([e[2] for e in edges_wt])/self.total_weight
        return edges[np.random.choice(len(edges), n, p=probs)]

    def evolve_hb(self, state, angle):
        c = np.cos(angle)
        s = 1j*np.sin(angle)
        for i in range(self.qubits):
            # The next 3 lines apply X to the ith qubit
            swapped = state.copy()
            swapped = swapped.reshape(state.size >> (i + 1), 1 << (i + 1))
            swapped = np.roll(swapped, 1 << i, axis=1).ravel()
            state *= c
            state -= s*swapped

    def evolve_hc(self, state, angle):
        state *= np.exp(-1j*angle*self.hc)

    def prepare_state(self, angles, init_state=None):
        if init_state is None:
            state = np.full(2**self.qubits, np.exp2(-self.qubits/2), dtype=complex)
        for gamma, beta in zip(*[iter(angles)]*2):
            self.evolve_hc(state, gamma)
            self.evolve_hb(state, beta)
        if len(angles) % 2 == 1:
            self.evolve_hc(state, angles[-1])
        return state

    def objective_state(self, state, exact=True, n=1):
        amps = np.real(state*np.conj(state))
        exact_value = -amps @ self.hc
        if exact:
            return exact_value
        else:
            p = -exact_value/self.total_weight
            return -(self.total_weight/n)*np.random.binomial(n, p)

    def objective(self, angles, exact=True, n=1):
        state = self.prepare_state(angles)
        return self.objective_state(state, exact, n)
