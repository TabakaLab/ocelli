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
    """Nearest neighbors search
    
    Computes exact or approximate nearest neighbors using sklearn and nmslib libraries, respectively. 
    We recommend sklearn until its runtime noticeably increases.

    Parameters
    ----------
    adata
        The annotated data matrix.
    x
        A list of `adata.obsm` keys storing embedding data of, e.g., modalities.
        If :obj:`None`, keys are loaded from `adata.uns["modalities"]`. (default: :obj:`None`)
    n_neighbors
        Number of nearest neighbors. (default: 20)
    method
        The method used for the neareast neighbor search. Valid options: `sklearn` 
        (uses the :class:`scikit-learn` package and computes exact nearest neighbors) 
        and `nmslib` (uses the :class:`nmslib` package, which is faster for very big datasets
        but less accurate (computes approximate nearest neighbors).
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    verbose
        Print progress notifications. (default: `False`)
    copy
        Return a copy of :class:`anndata.AnnData. (default: `False`)

    Returns
    -------
    :obj:`None`
        By default (`copy=False`), updates `adata` with the following fields:
        `adata.obsm[neighbors_*]` (arrays with nearest neighbor indices, `*` denotes modality name), 
        and `adata.obsm[distances_*]` (arrays with nearest neighbor distances, `*` denotes modality name).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
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
