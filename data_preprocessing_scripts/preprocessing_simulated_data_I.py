from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import anndata as ad
import warnings
from multiprocessing import cpu_count
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix
warnings.filterwarnings('ignore')


# utility functions
def downsample(X: np.ndarray, sample_size: int, n_neighbors=25, power=-3):
    neigh = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=cpu_count())
    neigh.fit(X)
    dists, _ = neigh.kneighbors(X)
    dists = np.power(dists[:, -1], power)
    probs = np.reciprocal(dists)
    probs = probs/np.sum(probs)
    
    return list(np.random.choice(range(X.shape[0]), sample_size, replace=False, p=probs))


def score(cell0, cell1, dir_vec):
    return 1 - cosine(cell1 - cell0, dir_vec)


def create_view(distributions,
                initial_population_size, 
                cells_per_type,
                filename=None, 
                n_neighbors=20,
                get_cosines=False,
                labels=None, 
                noise_batch=100):
    
    n_cells = cells_per_type * len(distributions)
    if labels is None:
        labels = [i//cells_per_type for i in range(n_cells)]

    cells = []
    for d in distributions:
        coords = np.random.multivariate_normal(d[0], d[1], initial_population_size)
        coords = coords[downsample(coords, sample_size=cells_per_type)]
        sorted_ids = np.argsort([np.linalg.norm(coord - d[2]) for coord in coords])
        coords = coords[sorted_ids, :]
        cells.append(coords)
    cells = np.concatenate(cells)
    
    noise = [np.random.permutation(split) for split in np.array_split(range(n_cells), n_cells // noise_batch)]
    noise = np.concatenate(noise)
    cells = cells[noise, :]
    
    if get_cosines:
        neigh = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=cpu_count())
        neigh.fit(cells)
        _, nn_ids = neigh.kneighbors(cells)
 
        vals, ids0, ids1 = [], [], []
        for i, indices in enumerate(nn_ids):
            d_id = i // cells_per_type
            for j in indices:
                val = score(cells[i], cells[j], distributions[d_id][3])
                if val > 0:
                    vals.append(val)
                    ids0.append(i)
                    ids1.append(j)
                elif val < 0:
                    vals.append(0.01)
                    ids0.append(i)
                    ids1.append(j)

        directions = coo_matrix((vals, (ids0, ids1))).tocsr()
        
    if get_cosines:
        return np.asarray(cells), pd.DataFrame(labels, columns=['type']), directions
    else:
        return np.asarray(cells), pd.DataFrame(labels, columns=['type'])
    

# generate modalities
distributions0 = (([11, 0, 0], [[20, 0, 0], [0, 2, 0], [0, 0, 2]], [0, -2, 0], [1, 0, 0]),
                  ([0, 12, 0], [[2, 0, 0], [0, 20, 0], [0, 0, 2]], [0, -5, 0], [0, 1, 0]), 
                  ([0, 28, 0], [[2, 0, 0], [0, 3, 0], [0, 0, 2]], [0, 21, 0], [0, 1, 0]), 
                  ([0, 30, 0], [[2, 0, 0], [0, 5, 0], [0, 0, 2]], [0, 21, 0], [0, 1, 0]),
                  ([0, 38, 0], [[0, 1, 0], [1, 0, 0], [0, 0, 1]], [0, 33, 0], [0, 1, 0]),
                  ([0, 38, 0], [[0, 1, 0], [1, 0, 0], [0, 0, 1]], [0, 33, 0], [0, 1, 0]))
v0, l0, d0 = create_view(distributions0, n_neighbors=100, initial_population_size=5000, 
                         cells_per_type=1000, get_cosines=True, noise_batch=100)

distributions1 = (([5, 0, 0], [[0, 1, 0], [4, 0, 0], [0, 0, 1]], [15, 0, 0], [-1, 0, 0]),
                  ([0, 0, 0], [[0, 1, 0], [14, 0, 0], [0, 0, 1]], [15, 0, 0], [-1, 0, 0]), 
                  ([-10, 8, 0], [[0, 10, 0], [2, 0, 0], [0, 0, 2]], [-10, 0, 0], [0, 1, 0]), 
                  ([-10, -8, 0], [[0, 10, 0], [2, 0, 0], [0, 0, 2]], [-10, 0, 0], [0, 1, 0]),
                  ([-10, -14, 0], [[0, 1, 0], [1, 0, 0], [0, 0, 1]], [-10, -8, 0], [0, 1, 0]), 
                  ([-10, -14, 0], [[0, 1, 0], [1, 0, 0], [0, 0, 1]], [-10, -8, 0], [0, 1, 0]))
v1, l1, d1 = create_view(distributions1, n_neighbors=100, initial_population_size=5000, 
                         cells_per_type=1000, get_cosines=True, noise_batch=100)

distributions2 = (([0, 0, 0], [[0, 1, 0], [1, 0, 0], [0, 0, 1]], [0, -5, -5], [0, 1, 1]),
                  ([0, 0, 0], [[0, 1, 0], [1, 0, 0], [0, 0, 1]], [0, -5, -5], [0, 1, 1]), 
                  ([0, 0, 0], [[0, 1, 0], [1, 0, 0], [0, 0, 1]], [0, -5, -5], [0, 1, 1]), 
                  ([0, 3, 3], [[0, 1, 0], [1, 0, 0], [0, 0, 1]], [0, -2, -2], [0, 1, 1]),
                  ([0, 11, 4], [[0, 10, 0], [1, 0, 0], [0, 0, 1]], [0, 0, 4], [0, 1, 0]), 
                  ([0, 4, 11], [[0, 1, 0], [1, 0, 0], [0, 0, 10]], [0, 4, 0], [0, 0, 1]))
v2, l2, d2 = create_view(distributions2, n_neighbors=100, initial_population_size=5000, 
                         cells_per_type=1000, get_cosines=True, noise_batch=100)

# cell ordering
order = [i for i in range(1000)] + [i for i in range(1000)] + [i for i in range(1000, 2000)] + [
    i for i in range(1000, 2000)] + [i for i in range(2000, 3000)] + [i for i in range(2000, 3000)]


# create AnnData object
adata = ad.AnnData(csr_matrix((6000, 1)))
adata.obsm['modality0'] = v0
adata.obsm['modality1'] = v1
adata.obsm['modality2'] = v2
adata.uns['modalities'] = ['modality0', 'modality1', 'modality2']

pseudotime_max = np.max(order)

d_celltype = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}

df_obs = l0
df_obs['type'] = [d_celltype[v] for v in df_obs['type']]
df_obs['pseudotime'] = [v/pseudotime_max for v in order]

adata.obs = df_obs

adata.write('simulated_data_I.h5ad', compression='gzip', compression_opts=9)
