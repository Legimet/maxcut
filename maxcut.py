import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt

def x_gate(state, i):
    reshaped = state.reshape(state.size >> (i + 1), 1 << (i + 1))
    return np.roll(reshaped, 1 << i, axis=1).ravel()

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
            xstate = x_gate(state, i)
            state *= c
            state -= s*xstate

    def evolve_hc(self, state, angle):
        state *= np.exp(-1j*angle*self.hc)

    def prepare_state(self, angles, init_state=None, starthb=False):
        if init_state is None:
            state = np.full(len(self.hc), np.exp2(-self.qubits/2), dtype=complex)
        else:
            state = init_state.copy()
        for i in angles:
            if starthb:
                self.evolve_hb(state, i)
            else:
                self.evolve_hc(state, i)
            starthb = not starthb
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

    def gradient(self, angles, indices, return_obj=False):
        ret = np.empty(len(indices))
        idx, inv = np.unique(indices, return_inverse=True)
        idx = np.concatenate(([0], idx))
        inv += 1
        left = [None]*len(idx)
        derivs = np.empty(len(idx))
        for i in range(1, len(idx)):
            left[i] = self.prepare_state(angles[idx[i-1]:idx[i]], left[i-1], idx[i-1])
        left[0] = self.prepare_state(angles[idx[-1]:], left[-1], idx[-1])
        right = self.hc*left[0]
        for i in range(1, len(idx)):
            if idx[i] % 2 == 0:
                state = np.zeros_like(left[i])
                for j in range(self.qubits):
                    state += x_gate(left[i], j)
                left[i] = state
            else:
                left[i] *= self.hc
            left[i] = self.prepare_state(angles[idx[i]:], left[i], idx[i])
            derivs[i] = 2*np.imag(np.vdot(left[i], right))
        for i in range(len(ret)):
            ret[i] = derivs[inv[i]]
        if return_obj:
            return ret, -np.real(np.vdot(left[0], right))
        else:
            return ret
