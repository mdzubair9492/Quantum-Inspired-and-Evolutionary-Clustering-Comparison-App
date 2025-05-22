
import numpy as np
import random
from typing import Tuple, Dict, Any
from sklearn.metrics import pairwise_distances_argmin_min, davies_bouldin_score, silhouette_score
from utils.metrics import compute_metrics

class QBitIndividual:
    def __init__(self, num_qbits: int, max_clusters: int):
        
        self.qbits = np.array([[1/np.sqrt(2), 1/np.sqrt(2)] for _ in range(num_qbits)])
        
        self.length = random.randint(1, max_clusters) if max_clusters >= 1 else num_qbits

    def measure(self) -> list:
        
        bits = []
        for a, b in self.qbits:
            p0 = a**2
            bits.append(0 if random.random() < p0 else 1)
        measured = bits[:self.length]
        return measured if measured else [bits[0]]

    def rotate(self, best_bits: list, delta: float):
        
        for i, (a, b) in enumerate(self.qbits):
            target = best_bits[i] if i < len(best_bits) else 0
            current = 0 if a**2 >= 0.5 else 1
            if current != target:
                new_a = np.cos(delta)*a - np.sin(delta)*b
                new_b = np.sin(delta)*a + np.cos(delta)*b
                norm = np.hypot(new_a, new_b)
                self.qbits[i] = [new_a/norm, new_b/norm]

    def apply_genetic_ops(self,
                          other: 'QBitIndividual',
                          crossover_rate: float,
                          mutation_rate: float) -> 'QBitIndividual':
       
        if len(self.qbits) > 1 and random.random() < crossover_rate:
            pt = random.randint(1, len(self.qbits) - 1)
            child_qbits = np.vstack((self.qbits[:pt], other.qbits[pt:]))
        else:
            child_qbits = self.qbits.copy()

        
        for i in range(len(child_qbits)):
            if random.random() < mutation_rate:
                child_qbits[i] = child_qbits[i][::-1]

       
        child = QBitIndividual.__new__(QBitIndividual)
        child.qbits = child_qbits
        child.length = random.choice([self.length, other.length])
        return child

class AQGUK:
    def __init__(self,
                 data: np.ndarray,
                 pop_size: int = 20,
                 max_iters: int = 50,
                 max_k: int = 10,
                 rotation_delta: float = 0.01,
                 crossover_rate: float = 0.7,
                 mutation_rate: float = 0.01):
        self.data = data
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.max_k = max_k
        self.delta = rotation_delta
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        self.population = [QBitIndividual(self.max_k, self.max_k) for _ in range(self.pop_size)]

    def decode(self, ind: QBitIndividual) -> Tuple[np.ndarray, int]:
        bits = ind.measure()
        unique_bits = list(dict.fromkeys(bits))
        k = max(1, len(unique_bits))
        centroids = self.data[:k].copy()
        return centroids, k

    def evaluate(self, centroids: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        labels, dists = pairwise_distances_argmin_min(self.data, centroids)
        sse = np.sum(dists**2)
        sil = silhouette_score(self.data, labels) if len(np.unique(labels)) >= 2 else -1.0
        db = davies_bouldin_score(self.data, labels) if len(np.unique(labels)) >= 2 else np.inf
        return labels, sse, sil, db

    def run(self) -> Tuple[np.ndarray, int, float]:
        best_db = np.inf
        best_bits = None
        best_centroids = None
        best_labels = None

        for _ in range(self.max_iters):
            evaluations = []
            for ind in self.population:
                cents, _ = self.decode(ind)
                labels, sse, sil, db = self.evaluate(cents)
                evaluations.append((ind, cents, labels, db))
                if db < best_db:
                    best_db = db
                    best_bits = ind.measure()
                    best_centroids = cents
                    best_labels = labels

            
            for ind, _, _, _ in evaluations:
                ind.rotate(best_bits, self.delta)

            
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1, p2 = random.sample(self.population, 2)
                child = p1.apply_genetic_ops(p2, self.crossover_rate, self.mutation_rate)
                new_pop.append(child)
            self.population = new_pop

        return best_labels, best_centroids.shape[0], best_db

def fit_predict(data: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    pop_size  = params.get('population_size', 20)
    max_iters = params.get('max_generations', 50)
    max_k     = params.get('max_k', 10)
    delta     = params.get('rotation_delta', 0.01)
    cx_rate   = params.get('crossover_rate', 0.7)
    mut_rate  = params.get('mutation_rate', 0.01)

    aq = AQGUK(data,
              pop_size=pop_size,
              max_iters=max_iters,
              max_k=max_k,
              rotation_delta=delta,
              crossover_rate=cx_rate,
              mutation_rate=mut_rate)
    labels, best_k, best_db = aq.run()

   
    unique = np.unique(labels)
    cents = np.array([data[labels == k].mean(axis=0) for k in unique])
    _, dists = pairwise_distances_argmin_min(data, cents)
    sse = np.sum(dists**2)
    sil = silhouette_score(data, labels) if len(unique) >= 2 else -1.0

    metrics = compute_metrics(data, labels, sse)
    metrics.update({
        'davies_bouldin_used': best_db,
        'silhouette_score': sil,
        'n_clusters': best_k
    })
    return labels, metrics

