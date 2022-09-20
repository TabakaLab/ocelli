import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from multiprocessing import cpu_count
import ray
from anndata import AnnData


def gaussian_kernel(x, y, epsilon_x, epsilon_y):
    """Adjusted Gaussian kernel"""
    epsilons_mul = epsilon_x * epsilon_y
    if epsilons_mul == 0:
        return 0
    else:
        try:
            x, y = x.toarray(), y.toarray()
        except AttributeError:
            pass

        x, y = x.flatten(), y.flatten()

        return np.exp(-1 * np.power(np.linalg.norm(x - y), 2) / epsilons_mul)


class MultiViewDiffMaps():
    """Multi-view diffusion maps class"""
    def __init__(self,
                 n_jobs=cpu_count()):
        if not ray.is_initialized():
            ray.init(num_cpus=n_jobs)
        self.n_jobs = n_jobs
        
    @staticmethod
    @ray.remote
    def __affinity_worker(views, nn, epsilons, weights, split, view_id, split_id):
        affinities = list()
        for cell0 in split[split_id]:
            for cell1 in nn[view_id][cell0]:
                if cell0 in nn[view_id][cell1] and cell0 < cell1:
                    pass
                else:
                    affinity = gaussian_kernel(views[view_id][cell0],
                                               views[view_id][cell1],
                                               epsilons[view_id][cell0],
                                               epsilons[view_id][cell1])
                    
                    mean_weight = (weights[cell0, view_id] + weights[cell1, view_id]) / 2
                    affinities.append([cell0, cell1, affinity * mean_weight])
                    affinities.append([cell1, cell0, affinity * mean_weight])
                    
        return np.asarray(affinities)
    
    def fit_transform(self, 
                      views, 
                      n_comps,
                      nn, 
                      epsilons, 
                      weights,
                      normalize_single_views,
                      eigval_times_eigvec,
                      random_state, 
                      verbose):
        
        n_views = len(views)
        n_cells = views[0].shape[0]

        split = np.array_split(range(n_cells), self.n_jobs)
        
        views_ref = ray.put(views)
        nn_ref = ray.put(nn)
        epsilons_ref = ray.put(epsilons)
        split_ref = ray.put(split)
        weights_ref = ray.put(weights)
        
        for view_id, _ in enumerate(views):
            affinities = [self.__affinity_worker.remote(views_ref, 
                                                        nn_ref, 
                                                        epsilons_ref,
                                                        weights_ref,
                                                        split_ref, 
                                                        view_id, 
                                                        i) 
                          for i in range(self.n_jobs)]

            affinities = ray.get(affinities)
            affinities = np.vstack(affinities)
            affinity_view = coo_matrix((affinities[:, 2], (affinities[:, 0], affinities[:, 1])), 
                                         shape=(n_cells, n_cells)).tocsr()
            
            if normalize_single_views:
                diag_vals = np.asarray([1 / val if val != 0 else 0 for val in affinity_view.sum(axis=1).A1])
                affinity_view = diags(diag_vals) @ affinity_view
            
            if view_id == 0:
                affinity_matrix = affinity_view
            else:
                affinity_matrix += affinity_view
                
            if verbose:
                print('Unimodal Markov chain calculated ({}/{})'.format(view_id, len(views)))
                
        diag_vals = np.asarray([1 / val if val != 0 else 1 for val in affinity_matrix.sum(axis=1).A1])
        affinity_matrix = diags(diag_vals) @ affinity_matrix
        
        affinity_matrix = affinity_matrix + affinity_matrix.T
        
        if verbose:
                print('Multimodal Markov chain calculated')
        
        if random_state is not None:
            np.random.seed(random_state)
            v0 = np.random.normal(0, 1, size=(affinity_matrix.shape[0]))
            eigvals, eigvecs = eigsh(affinity_matrix, k=n_comps + 11, which='LA', maxiter=100000, v0=v0)
        else:
            eigvals, eigvecs = eigsh(affinity_matrix, k=n_comps + 11, which='LA', maxiter=100000)
        eigvals, eigvecs = np.flip(eigvals)[1:n_comps + 1], np.flip(eigvecs, axis=1)[:, 1:n_comps + 1]
        rep = eigvecs * eigvals if eigval_times_eigvec else eigvecs
        
        if verbose:
            print('Eigendecomposition finished.')
    
        return rep


def MVDM(adata: AnnData,
         n_components: int = 10,
         views = None,
         weights_key: str = 'weights',
         neighbors_key: str = 'neighbors',
         epsilons_key: str = 'epsilons',
         output_key: str = 'X_mvdm',
         normalize_single_views: bool = True,
         eigval_times_eigvec: bool = True,
         n_jobs: int = -1,
         random_state = None,
         verbose: bool = False,
         copy: bool = False):
    """Multi-view diffusion maps

    Algorithm calculates multi-view diffusion maps cell embeddings using
    pre-calculated multi-view weights.

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_components
        The number of multi-view diffusion maps components. (default: 10)
    views
        A list of ``adata.obsm`` keys storing modalities.
        If :obj:`None`, views' keys are loaded from ``adata.uns['key_views']``. (default: :obj:`None`)
    weights_key
        ``adata.obsm[weights_key]`` stores multimodal weights. (default: ``weights``)
    neighbors_key
        ``adata.uns[neighbors_key]`` stores the nearest neighbor indices 
        (:class:`numpy.ndarray` of shape ``(n_views, n_cells, n_neighbors)``).
        (default: ``neighbors``)
    epsilons_key
        ``adata.uns[epsilons_key]`` stores epsilons used for adjusted Gaussian kernel 
        (:class:`numpy.ndarray` of shape ``(n_views, n_cells, n_neighbors)``).
        (default: ``epsilons``)
    output_key
        Multi-view diffusion maps embedding are saved to ``adata.obsm[output_key]``. (default: ``X_mvdm``)
    normalize_single_views
        If ``True``, single-view kernel matrices are normalized. (default: ``True``)
    eigval_times_eigvec
        If ``True``, the multi-view diffusion maps embedding is calculated by multiplying
        eigenvectors by eigenvalues. Otherwise, the embedding consists of unmultiplied eigenvectors. (default: ``True``)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    random_state
        Pass an :obj:`int` for reproducible results across multiple function calls. (default: :obj:`None`)
    verbose
        Print progress notifications. (default: ``False``)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obsm[output_key]`` (:class:`numpy.ndarray`).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """

    if views is None:
        if 'views' not in list(adata.uns.keys()):
            raise(NameError('No view keys found in adata.uns[views].'))
        views = adata.uns['views']
 
    if len(views) == 0:
        raise(NameError('No view keys found in adata.uns[views].'))

    if weights_key not in adata.obsm:
        raise(KeyError('No weights found in adata.uns["{}"]. Run ocelli.tl.weights.'.format(weights_key)))

    if neighbors_key not in adata.uns:
        raise (KeyError(
            'No nearest neighbors found in adata.uns["{}"]. Run ocelli.pp.neighbors.'.format(neighbors_key)))
        
    if epsilons_key not in adata.uns:
        raise(KeyError('No epsilons found in adata.uns["{}"]. Run ocelli.pp.neighbors.'.format(epsilons_key)))

    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])

    adata.obsm[output_key] = MultiViewDiffMaps(n_jobs).fit_transform(views = [adata.obsm[key] for key in views],
                                                                     n_components = n_components,
                                                                     nn = adata.uns[neighbors_key],
                                                                     epsilons = adata.uns[epsilons_key],
                                                                     weights = np.asarray(adata.obsm[weights_key]),
                                                                     normalize_single_views = normalize_single_views,
                                                                     eigval_times_eigvec = eigval_times_eigvec,
                                                                     random_state = random_state,
                                                                     verbose = verbose)

    if verbose:
        print('{} multi-view diffusion maps components calculated.'.format(n_components))
    
    return adata if copy else None
