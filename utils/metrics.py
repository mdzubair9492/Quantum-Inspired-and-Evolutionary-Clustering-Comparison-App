import time
from sklearn.metrics import silhouette_score, davies_bouldin_score

def compute_metrics(data, labels, inertia):
    start = time.time()
    sil = silhouette_score(data, labels)
    db = davies_bouldin_score(data, labels)
    return { 'inertia': inertia, 'silhouette': sil,
             'davies_bouldin': db, 'runtime': time.time()-start }