import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.sparse import coo_matrix, vstack, csr_matrix
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors
import anndata as ad
import warnings
from multiprocessing import cpu_count
from scipy.spatial.distance import cosine
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
    

def sort_celltype(cells):
    cells = np.concatenate(cells, axis=0)
    sorted_ids = np.concatenate([np.argsort([np.linalg.norm(cell - [10, 0, 0]) for cell in cells])])
    
    return cells[sorted_ids, :]


# generate modalities
distributions0 = (([13, 3, 0], [[10, 0, 0], [0, 1, 0], [0, 0, 1]], [3, 3, 0], [1, 0, 0]),
                  ([3, 13, 0], [[1, 0, 0], [0, 10, 0], [0, 0, 1]], [3, 3, 0], [0, 1, 0]),
                  ([-3, 0, 13], [[1, 0, 0], [0, 1, 0], [0, 0, 10]], [-3, 0, 3], [0, 0, 1]), 
                  ([-13, 0, 3], [[10, 0, 0], [0, 1, 0], [0, 0, 1]], [-3, 0, 3], [-1, 0, 0]),
                  ([0, -13, -3], [[1, 0, 0], [0, 10, 0], [0, 0, 1]], [0, -3, -3], [0, -1, 0]),
                  ([0, -3, -13], [[1, 0, 0], [0, 1, 0], [0, 0, 10]], [0, -3, -3], [0, 0, -1]), 
                  ([0, 0, 0], [[2, 0, 0], [0, 2, 0], [0, 0, 2]], [-8, -8, 0], [0, 0, 0]), 
                  ([0, 0, 0], [[2, 0, 0], [0, 2, 0], [0, 0, 2]], [8, 0, -8], [0, 0, 0]),
                  ([0, 0, 0], [[2, 0, 0], [0, 2, 0], [0, 0, 2]], [0, 8, 8], [0, 0, 0]),)

v1, l1, d1 = create_view(distributions0,
                         n_neighbors=100, 
                         initial_population_size=5000, 
                         cells_per_type=500, 
                         get_cosines=True, 
                         labels=[i // 500 for i in range(4500)])

initial_population_size = 5000
n_celltypes = 3
cells_per_type = 500
center_size = int(cells_per_type*0.5)
labels = [i // 1000 for i in range(3000)]
subs = [range(k*cells_per_type, (k+1)*cells_per_type) for k in range(n_celltypes)]

uniform = np.random.uniform(size=cells_per_type)
cellsA = np.random.multivariate_normal([22, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], initial_population_size)
cellsA = cellsA[downsample(cellsA, sample_size=cells_per_type)]
cellsA = sort_celltype([cellsA])

cellsB = np.random.multivariate_normal([22, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], initial_population_size)
cellsB = cellsB[downsample(cellsB, sample_size=cells_per_type)]
cellsB = sort_celltype([cellsB])

cellsC = np.random.multivariate_normal([3, -12, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], initial_population_size)
cellsC = cellsC[downsample(cellsC, sample_size=cells_per_type)]
cellsC = sort_celltype([cellsC])

cellsD = np.random.multivariate_normal([3, -12, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], initial_population_size)
cellsD = cellsD[downsample(cellsD, sample_size=cells_per_type)]
cellsD = sort_celltype([cellsD])

cellsE = np.random.multivariate_normal([3, 12, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], initial_population_size)
cellsE = cellsE[downsample(cellsE, sample_size=cells_per_type)]
cellsE = sort_celltype([cellsE])

cellsF = np.random.multivariate_normal([3, 12, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], initial_population_size)
cellsF = cellsF[downsample(cellsF, sample_size=cells_per_type)]
cellsF = sort_celltype([cellsF])

cellsG = np.expand_dims(uniform, 1) @ np.expand_dims(np.asarray([-10, 0, 0]), 0) + [20, 0, 0]
cellsG = sort_celltype([cellsG])

cellsH = np.expand_dims(uniform, 1) @ np.expand_dims(np.asarray([6, 10, 0]), 0) + [4, -10, 0]
cellsH = sort_celltype([cellsH])

cellsI = np.expand_dims(uniform, 1) @ np.expand_dims(np.asarray([6, -10, 0]), 0) + [4, 10, 0]
cellsI = sort_celltype([cellsI])

cells = np.concatenate((cellsA, cellsB, cellsC, cellsD, cellsE, cellsF, cellsG, cellsH, cellsI))

n_cells=4500
noise = [np.random.permutation(split) for split in np.array_split(range(n_cells), n_cells // 100)]
noise = np.concatenate(noise)

cells = cells[noise, :]

neigh = NearestNeighbors(n_neighbors=50, n_jobs=cpu_count())
neigh.fit(cells)
_, nn_ids = neigh.kneighbors(cells)

vals, ids0, ids1 = [], [], []
for i, indices in enumerate(nn_ids):
    for j in indices:
        val = score(cells[i], cells[j], [10, 0, 0])
        if val > 0:
            vals.append(val)
            ids0.append(i)
            ids1.append(j)
        elif val < 0:
            vals.append(0.01)
            ids0.append(i)
            ids1.append(j)
        
d0 = coo_matrix((vals, (ids0, ids1))).tocsr()
v0 = np.asarray(cells)

labels = [i//500 for i in range(4500)]

# cell ordering
order = [i for i in range(500, 1000)] + [i for i in range(500, 1000)] + [i for i in range(500, 1000)] + [
    i for i in range(500, 1000)] + [i for i in range(500, 1000)] + [i for i in range(500, 1000)] + [
    i for i in range(500)] + [i for i in range(500)] + [i for i in range(500)]

# create AnnData object
adata = ad.AnnData(csr_matrix((4500, 1)))
adata.obsm['modality0'] = v0
adata.obsm['modality1'] = v1
adata.uns['modalities'] = ['modality0', 'modality1']

pseudotime_max = np.max(order)

d_celltype = {0: 'D', 1: 'E', 2: 'H', 3: 'I', 4: 'F', 5: 'G', 6: 'A', 7: 'C', 8: 'B'}

df_obs = l1
df_obs['type'] = [d_celltype[v] for v in df_obs['type']]
df_obs['pseudotime'] = [v/pseudotime_max for v in order]

adata.obs = df_obs

adata.write('simulated_data_II.h5ad', compression='gzip', compression_opts=9)
