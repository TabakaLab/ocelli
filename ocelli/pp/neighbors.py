import nmslib
import numpy as np
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
from multiprocessing import cpu_count
import anndata as ad

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def NMSLIB(M, n_neighbors, n_jobs):
    if issparse(M):
        p = nmslib.init(method='hnsw', 
                    space='l2_sparse', 
                    data_type=nmslib.DataType.SPARSE_VECTOR, 
                    dtype=nmslib.DistType.FLOAT)
    else:
        p = nmslib.init(method='hnsw', 
                        space='l2')
        
    p.addDataPointBatch(M)
    p.createIndex({'M': 10, 'indexThreadQty': n_jobs, 'efConstruction': 100, 'post': 0, 'skip_optimized_index': 1})
    p.setQueryTimeParams({'efSearch': 100})
    output = p.knnQueryBatch(M, 
                             k=n_neighbors + 1, 
                             num_threads=n_jobs)
    labels, distances = list(), list()
    for record in output:
        labels.append(record[0][1:])
        distances.append(record[1][1:])
        
    labels, distances = np.stack(labels), np.stack(distances)
    
    return labels, distances


def neighbors(adata: ad.AnnData,
              x: list = None,
              n_neighbors: int = 20,
              method: str = 'sklearn',
              n_jobs: int = -1,
              verbose: bool = False,
              copy: bool = False):
    """
    Nearest neighbors search

    Computes exact or approximate nearest neighbors using sklearn or nmslib libraries. 
    The sklearn method calculates exact neighbors, while nmslib is faster and approximates 
    neighbors for large datasets.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param x: A list of keys in `adata.obsm` specifying embeddings to calculate neighbors for.
        If None, keys are loaded from `adata.uns["modalities"]`. (default: None)
    :type x: list or None

    :param n_neighbors: Number of nearest neighbors to compute. (default: 20)
    :type n_neighbors: int

    :param method: Method to compute nearest neighbors. Valid options are `sklearn` (exact)
        and `nmslib` (approximate). (default: 'sklearn')
    :type method: str

    :param n_jobs: Number of parallel jobs to use. If -1, all CPUs are used. (default: -1)
    :type n_jobs: int

    :param verbose: Whether to print progress notifications. (default: False)
    :type verbose: bool

    :param copy: Whether to return a copy of the AnnData object. If False, the input object
        is updated in-place. (default: False)
    :type copy: bool

    :returns: By default (`copy=False`), updates `adata` with nearest neighbor indices 
        and distances stored in `adata.obsm`. If `copy=True`, returns a copy of `adata`.
    :rtype: anndata.AnnData or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Example AnnData object
            adata = AnnData(X=np.random.random((100, 50)))
            adata.obsm['modality1'] = np.random.random((100, 10))

            # Compute neighbors
            oci.pp.neighbors(adata, x=['modality1'], n_neighbors=15, method='nmslib', n_jobs=4)
    """

    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])

    if x is None:
        if 'modalities' not in list(adata.uns.keys()) or len(adata.uns['modalities']) == 0:
            raise(NameError('No data found in adata.uns["modalities"].'))
        x = adata.uns['modalities']
 
    indices, distances = list(), list()

    for m in x:
        if method == 'sklearn':
            neigh = NearestNeighbors(n_neighbors=n_neighbors+1, n_jobs=n_jobs)
            neigh.fit(adata.obsm[m])
            nn = neigh.kneighbors(adata.obsm[m])
            
            adata.obsm['neighbors_{}'.format(m)] = nn[1][:, 1:]
            adata.obsm['distances_{}'.format(m)] = nn[0][:, 1:]

        elif method == 'nmslib':
            try:
                neigh = NMSLIB(adata.obsm[m], n_neighbors, n_jobs)
            except ValueError:
                raise(ValueError('The value n_neighbors={} is too high for NMSLIB. Practically, 20-50 neighbors are usually enough.'.format(n_neighbors)))
                
            adata.obsm['neighbors_{}'.format(m)] = neigh[0]
            adata.obsm['distances_{}'.format(m)] = neigh[1]
            
        else:
            raise(NameError('Wrong nearest neighbor search method. Valid options: sklearn, nmslib.'))
            
        if verbose:
                print('[{}]\t{} nearest neighbors calculated.'.format(m, n_neighbors))
    
    return adata if copy else None
