import nmslib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from multiprocessing import cpu_count
import anndata
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def nmslib_nn(M, n_neighbors, n_jobs):
    """NMSLIB nearest neighbors search"""
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


def neighbors(adata: anndata.AnnData,
              n_neighbors: int = 20,
              views = None,
              method: str = 'sklearn',
              neighbors_key: str = 'neighbors',
              epsilons_key: str = 'epsilons',
              distances_key: str = 'distances',
              n_jobs: int = -1,
              verbose: bool = False,
              copy: bool = False):
    """Nearest neighbors search for all modalities
    
    Two  nearest neighbors search methods are available: ``sklearn``, and ``nmslib``. 
    Both can run on dense and sparse arrays.
    
    ``sklearn`` method uses the :class:`scikit-learn` package and is teh default.
    We recommend ``sklearn`` until its runtime noticeably increases. ``nmslib`` method uses the :class:`nmslib` package,
    which is faster for very big datasets (hundreds of thousands of cells) but less accurate,
    as it is an approximate nearest neighbors algorithm.

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_neighbors
        The number of nearest neighbors. (default: 20)
    views
        A list of ``adata.obsm`` keys storing modalities.
        If :obj:`None`, views' keys are loaded from ``adata.uns['key_views']``. (default: :obj:`None`)
    method
        The method used for the neareast neighbor search.
        Possible options: ``sklearn``, ``nmslib``. (default: ``sklearn``)
    neighbors_key
        The nearest neighbors indices are saved in ``adata.uns[neighbors_key]``.
        (default: ``neighbors``)
    epsilons_key
        The nearest neighbors epsilons are saved in ``adata.uns[epsilons_key]``.
        (default: ``epsilons``)
    distances_key
        The nearest neighbors distances are saved in ``adata.uns[distances_key]``.
        (default: ``distances``)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    verbose
        Print progress notifications. 
    copy
        Return a copy of :class:`anndata.AnnData. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.uns[neighbors_key]`` (:class:`numpy.ndarray` of shape ``(n_views, n_cells, n_neighbors)``),
        ``adata.uns[epsilons_key]`` (:class:`numpy.ndarray` of shape ``(n_views, n_cells, n_neighbors)``),
        ``adata.uns[distances_key]`` (:class:`numpy.ndarray` of shape ``(n_views, n_cells)``).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """

    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])

    if views is None:
        if 'views' not in list(adata.uns.keys()) or len(adata.uns['views']) == 0:
            raise(NameError('No view keys found in adata.uns["views"].'))
        views = adata.uns['views']
 
    indices, distances, epsilons = list(), list(), list()
    epsilons_thr = min([n_neighbors, 20]) - 1

    for v in views:
        if method == 'sklearn':
            neigh = NearestNeighbors(n_neighbors=n_neighbors+1, n_jobs=n_jobs)
            neigh.fit(adata.obsm[v])
            nn = neigh.kneighbors(adata.obsm[v])
            
            indices.append(nn[1][:, 1:])
            distances.append(nn[0][:, 1:])
            epsilons.append(nn[0][:, epsilons_thr + 1])
            
        elif method == 'nmslib':
            try:
                neigh = nmslib_nn(adata.obsm[v], n_neighbors, n_jobs)
            except ValueError:
                raise(ValueError('The value n_neighbors={} is too high for NMSLIB. Practically, 20-50 neighbors are almost always enough.'.format(n_neighbors)))
                
            indices.append(neigh[0])
            distances.append(neigh[1])
            epsilons.append(neigh[1][:, epsilons_thr])
            
        else:
            raise(NameError('Wrong nearest neighbor search method. Choose one from: sklearn, nmslib.'))

    adata.uns[neighbors_key] = np.asarray(indices)
    adata.uns[distances_key] = np.asarray(distances)
    adata.uns[epsilons_key] = np.asarray(epsilons)

    if verbose:
        print('{} nearest neighbors calculated.'.format(n_neighbors))

    return adata if copy else None
