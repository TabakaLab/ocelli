import nmslib
import numpy as np
import multiprocessing as mp
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from multiprocessing import cpu_count
from anndata import AnnData
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def nmslib_dense(M,
                 n_neighbors,
                 num_threads=mp.cpu_count()):
    """NMSLIB nearest neighbors search for dense matrices"""
    p = nmslib.init(method='hnsw', 
                    space='l2')
    p.addDataPointBatch(M)
    p.createIndex({'M': 10, 'indexThreadQty': num_threads, 'efConstruction': 100, 'post': 0, 'skip_optimized_index': 1})
    p.setQueryTimeParams({'efSearch': 100})
    output = p.knnQueryBatch(M, 
                             k=n_neighbors + 1, 
                             num_threads=num_threads)
    labels, distances = list(), list()
    for record in output:
        labels.append(record[0][1:])
        distances.append(record[1][1:])
        
    labels, distances = np.stack(labels), np.stack(distances)
    
    return labels, distances


def nmslib_sparse(M: csr_matrix,
                  n_neighbors,
                  num_threads=mp.cpu_count()):
    """NMSLIB nearest neighbors search for sparse matrices"""
    assert(isinstance(M, csr_matrix))
    
    p = nmslib.init(method='hnsw', 
                    space='l2_sparse', 
                    data_type=nmslib.DataType.SPARSE_VECTOR, 
                    dtype=nmslib.DistType.FLOAT)
    p.addDataPointBatch(M)
    p.createIndex({'M': 10, 'indexThreadQty': num_threads, 'efConstruction': 100, 'post': 0, 'skip_optimized_index': 1})
    p.setQueryTimeParams({'efSearch': 100})
    output = p.knnQueryBatch(M, 
                             k=n_neighbors + 1, 
                             num_threads=num_threads)
    labels, distances = list(), list()
    for record in output:
        labels.append(record[0][1:])
        distances.append(record[1][1:])
    labels, distances = np.asarray(labels), np.asarray(distances)
    
    return labels, distances


def neighbors(adata: AnnData,
              n_neighbors: int = 20,
              view_keys = None,
              mode: str = 'sklearn',
              neighbors_key: str = 'neighbors',
              epsilons_key: str = 'epsilons',
              distances_key: str = 'distances',
              n_jobs: int = -1, 
              copy: bool = False):
    """Nearest neighbors search for all views
    
    Three modes are currently available: ``sklearn``, ``nmslib_dense``
    (both for dense arrays), and ``nmslib_sparse`` (for sparse arrays).
    
    The ``sklearn`` mode uses the :class:`scikit-learn` package.
    We recommend this method for dense arrays, when the number of cells, 
    or data dimensionality is not very high, or until method noticeably slows down.
    
    The ``nmslib_dense`` mode uses the :class:`nmslib` package,
    which is faster but less accurate, as it is an approximate nearest neighbors algorithm.
    When ``sklearn`` is too slow, switch to ``nmslib_dense``.
    
    The ``nmslib_sparse`` mode works for sparse arrays of type :class:`scipy.sparse.csr_matrix`.
    This method is more memory-efficient but is noticeably slower than nmslib_dense. 
    Hence, ``nmslib_sparse`` should be the third go-to option. If the memory allows,
    it is usually faster to convert a sparse matrix into a dense one and proceed
    with the ``nmslib_dense`` mode. Note that after preprocessing, views are almost always
    saved as dense :class:`numpy.ndarray` arrays.

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_neighbors
        The number of nearest neighbors. (default: 20)
    view_keys
        If None, view keys are loaded from ``adata.uns['key_views']``. Otherwise,
        ``view_keys`` should be a :class:`list` of ``adata.obsm`` keys,
        where views are stored. (default: :obj:`None`)
    mode
        The method used for the neareast neighbor search.
        Possible options: ``sklearn``, ``nmslib_dense``, ``nmslib_sparse``. (default: ``sklearn``)
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

    if view_keys is None:
        if 'view_keys' not in list(adata.uns.keys()):
            raise(NameError('No view keys found in adata.uns["view_keys"].'))
        view_keys = adata.uns['view_keys']
 
    if len(view_keys) == 0:
        raise(NameError('No view keys found in adata.uns["view_keys"].'))
    

    indices, distances, epsilons = list(), list(), list()
    epsilons_thr = min([n_neighbors, 20]) - 1

    if mode == 'sklearn':
        for v in view_keys:
            neigh = NearestNeighbors(n_neighbors=n_neighbors+1, n_jobs=n_jobs)
            neigh.fit(adata.obsm[v])
            nn = neigh.kneighbors(adata.obsm[v])
            indices.append(nn[1][:, 1:])
            distances.append(nn[0][:, 1:])
            epsilons.append(nn[0][:, epsilons_thr + 1])
            
    elif mode == 'nmslib_dense':
        for v in view_keys:
            try:
                neigh = nmslib_dense(adata.obsm[v], n_neighbors, n_jobs)
                indices.append(neigh[0])
                distances.append(neigh[1])
                epsilons.append(neigh[1][:, epsilons_thr])
            except ValueError:
                raise(ValueError('The value n_neighbors={} is too high for NMSLIB. Practically, 20-50 neighbors are almost always enough.'.format(n_neighbors)))
            
    elif mode == 'nmslib_sparse':
        for v in view_keys:
            try:
                neigh = nmslib_sparse(adata.obsm[v], n_neighbors, n_jobs)
                indices.append(neigh[0])
                distances.append(neigh[1])
                epsilons.append(neigh[1][:, epsilons_thr])
            except ValueError:
                raise(ValueError('The value n_neighbors={} is too high for NMSLIB. Practically, 20-50 neighbors are almost always enough.'.format(n_neighbors)))
    else:
        raise(NameError('Wrong nearest neighbor search mode. Choose one from: sklearn, nmslib_dense, nmslib_sparse.'))

    adata.uns[neighbors_key] = np.asarray(indices)
    adata.uns[distances_key] = np.asarray(distances)
    adata.uns[epsilons_key] = np.asarray(epsilons)

    print('{} nearest neighbors calculated.'.format(n_neighbors))

    return adata if copy else None
