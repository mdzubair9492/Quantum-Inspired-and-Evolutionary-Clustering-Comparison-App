import numpy as np
from typing import Tuple, Dict, Any
from utils.metrics import compute_metrics

class Individual:
    def __init__(self, centroids: np.ndarray):
        self.centroids = centroids.copy()
        self.fitness = float('inf')

class GeneticKMeans:
    def __init__(
        self,
        n_clusters: int,
        population_size: int = 20,
        max_generations: int = 100,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.01
    ):
        if n_clusters <= 0:
            raise ValueError
        self.n_clusters = n_clusters
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population = []
        self.best = None

    def _init_pop(self, data: np.ndarray):
        n = data.shape[0]
        self.population = []
        for _ in range(self.population_size):
            idx = np.random.choice(n, self.n_clusters, replace=False)
            self.population.append(Individual(data[idx]))

    def _eval(self, ind: Individual, data: np.ndarray) -> float:
        d = np.linalg.norm(data[:,None,:] - ind.centroids[None,:,:], axis=2)
        labels = np.argmin(d, axis=1)
        return sum(np.sum((data[labels==k] - ind.centroids[k])**2) for k in range(self.n_clusters))

    def _tourn(self, k=3):
        pick = np.random.choice(self.population, k, replace=False)
        return min(pick, key=lambda x: x.fitness)

    def _cross(self, p1, p2):
        if np.random.rand() < self.crossover_prob:
            pt = np.random.randint(1, self.n_clusters)
            c = np.vstack((p1.centroids[:pt], p2.centroids[pt:]))
            return Individual(c)
        return Individual(p1.centroids)

    def _mut(self, ind, data):
        for i in range(self.n_clusters):
            if np.random.rand() < self.mutation_prob:
                ind.centroids[i] = data[np.random.randint(data.shape[0])]

    def fit_predict(self, data: np.ndarray, params: Dict[str,Any]) -> Tuple[np.ndarray,Dict[str,Any]]:
        self.population_size = params.get('population_size', self.population_size)
        self.max_generations = params.get('max_generations', self.max_generations)
        self.crossover_prob  = params.get('crossover_prob', self.crossover_prob)
        self.mutation_prob   = params.get('mutation_prob', self.mutation_prob)

        self._init_pop(data)
        for ind in self.population:
            ind.fitness = self._eval(ind, data)
        self.best = min(self.population, key=lambda x: x.fitness)

        for _ in range(self.max_generations):
            new = []
            while len(new) < self.population_size:
                p1, p2 = self._tourn(), self._tourn()
                child = self._cross(p1,p2)
                self._mut(child,data)
                child.fitness = self._eval(child,data)
                new.append(child)
            self.population = new
            gbest = min(self.population, key=lambda x: x.fitness)
            if gbest.fitness < self.best.fitness:
                self.best = gbest

        d = np.linalg.norm(data[:,None,:] - self.best.centroids[None,:,:], axis=2)
        labels = np.argmin(d, axis=1)
        metrics = compute_metrics(data, labels, self.best.fitness)
        return labels, metrics

def fit_predict(data: np.ndarray, params: dict):
    ga = GeneticKMeans(n_clusters=params['n_clusters'],
                       population_size=params.get('population_size',20),
                       max_generations=params.get('max_generations',100),
                       crossover_prob=params.get('crossover_prob',0.8),
                       mutation_prob=params.get('mutation_prob',0.01))
    return ga.fit_predict(data, params)






