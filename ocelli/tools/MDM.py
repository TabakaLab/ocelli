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


class MultimodalDiffusionMaps():
    """Multimodal diffusion maps class"""
    def __init__(self,
                 n_jobs=cpu_count()):
        if not ray.is_initialized():
            ray.init(num_cpus=n_jobs)
        self.n_jobs = n_jobs
        
    @staticmethod
    @ray.remote
    def __affinity_worker(modalities, nn, epsilons, weights, split, modality_id, split_id):
        affinities = list()
        for cell0 in split[split_id]:
            for cell1 in nn[modality_id][cell0]:
                if cell0 in nn[modality_id][cell1] and cell0 < cell1:
                    pass
                else:
                    affinity = gaussian_kernel(modalities[modality_id][cell0],
                                               modalities[modality_id][cell1],
                                               epsilons[modality_id][cell0],
                                               epsilons[modality_id][cell1])
                    
                    mean_weight = (weights[cell0, modality_id] + weights[cell1, modality_id]) / 2
                    affinities.append([cell0, cell1, affinity * mean_weight])
                    affinities.append([cell1, cell0, affinity * mean_weight])
                    
        return np.asarray(affinities)
    
    def fit_transform(self, 
                      modalities, 
                      n_comps,
                      nn, 
                      epsilons, 
                      weights,
                      unimodal_norm,
                      eigval_times_eigvec,
                      random_state, 
                      verbose):
        
        n_modalities = len(modalities)
        n_cells = modalities[0].shape[0]

        split = np.array_split(range(n_cells), self.n_jobs)
        
        modalities_ref = ray.put(modalities)
        nn_ref = ray.put(nn)
        epsilons_ref = ray.put(epsilons)
        split_ref = ray.put(split)
        weights_ref = ray.put(weights)
        
        for modality_id, _ in enumerate(modalities):
            affinities = [self.__affinity_worker.remote(modalities_ref, 
                                                        nn_ref, 
                                                        epsilons_ref,
                                                        weights_ref,
                                                        split_ref, 
                                                        modality_id, 
                                                        i) 
                          for i in range(self.n_jobs)]

            affinities = ray.get(affinities)
            affinities = np.vstack(affinities)
            affinity_modality = coo_matrix((affinities[:, 2], (affinities[:, 0], affinities[:, 1])), 
                                         shape=(n_cells, n_cells)).tocsr()
            
            if unimodal_norm:
                diag_vals = np.asarray([1 / val if val != 0 else 0 for val in affinity_modality.sum(axis=1).A1])
                affinity_modality = diags(diag_vals) @ affinity_modality
            
            if modality_id == 0:
                affinity_matrix = affinity_modality
            else:
                affinity_matrix += affinity_modality
                
            if verbose:
                print('Unimodal Markov chain calculated ({}/{})'.format(modality_id + 1, len(modalities)))
                
        diag_vals = np.asarray([1 / val if val != 0 else 1 for val in affinity_matrix.sum(axis=1).A1])
        affinity_matrix = diags(diag_vals) @ affinity_matrix
        
        affinity_matrix = affinity_matrix + affinity_matrix.T
        
        if verbose:
                print('Multimodal Markov chain calculated')
        
        if random_state is not None:
            if random_state == 0:
                v0 = np.zeros(affinity_matrix.shape[0])
            else:
                np.random.seed(random_state)
                v0 = np.random.normal(0, 1, size=(affinity_matrix.shape[0]))
        else:
            v0 = None
        eigvals, eigvecs = eigsh(affinity_matrix, k=n_comps + 11, which='LA', maxiter=100000, v0=v0)
        eigvals, eigvecs = np.flip(eigvals)[1:n_comps + 1], np.flip(eigvecs, axis=1)[:, 1:n_comps + 1]
        rep = eigvecs * eigvals if eigval_times_eigvec else eigvecs
        
        if verbose:
            print('Eigendecomposition finished.')
    
        return rep


def MDM(adata: AnnData,
         n_components: int = 10,
         modalities = None,
         weights_key: str = 'weights',
         neighbors_key: str = 'neighbors',
         epsilons_key: str = 'epsilons',
         output_key: str = 'X_mdm',
         unimodal_norm: bool = True,
         eigval_times_eigvec: bool = True,
         n_jobs: int = -1,
         random_state = None,
         verbose: bool = False,
         copy: bool = False):
    """Multimodal Diffusion Maps

    Algorithm calculates multimodal diffusion maps cell embeddings using
    pre-calculated multimodal weights.

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_components
        The number of multimodal diffusion maps components. (default: 10)
    modalities
        A list of ``adata.obsm`` keys storing modalities.
        If :obj:`None`, modalities' keys are loaded from ``adata.uns[modalities]``. (default: :obj:`None`)
    weights_key
        ``adata.obsm[weights_key]`` stores multimodal weights. (default: `weights`)
    neighbors_key
        ``adata.uns[neighbors_key]`` stores the nearest neighbor indices 
        (:class:`numpy.ndarray` of shape ``(n_modalities, n_cells, n_neighbors)``).
        (default: ``neighbors``)
    epsilons_key
        ``adata.uns[epsilons_key]`` stores epsilons used for adjusted Gaussian kernel 
        (:class:`numpy.ndarray` of shape ``(n_modalities, n_cells, n_neighbors)``).
        (default: ``epsilons``)
    output_key
        Multimodal diffusion maps embedding is saved to ``adata.obsm[output_key]``. (default: `X_mdm`)
    unimodal_norm
        If ``True``, unimodal kernel matrices are normalized. (default: ``True``)
    eigval_times_eigvec
        If ``True``, the multimodal diffusion maps embedding is calculated by multiplying
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

    if modalities is None:
        if 'modalities' not in list(adata.uns.keys()):
            raise(NameError('No modality keys found in adata.uns[modalities].'))
        modalities = adata.uns['modalities']
 
    if len(modalities) == 0:
        raise(NameError('No modality keys found in adata.uns[modalities].'))

    if weights_key not in adata.obsm:
        raise(KeyError('No weights found in adata.uns["{}"]. Run ocelli.tl.weights.'.format(weights_key)))

    if neighbors_key not in adata.uns:
        raise (KeyError(
            'No nearest neighbors found in adata.uns["{}"]. Run ocelli.pp.neighbors.'.format(neighbors_key)))
        
    if epsilons_key not in adata.uns:
        raise(KeyError('No epsilons found in adata.uns["{}"]. Run ocelli.pp.neighbors.'.format(epsilons_key)))

    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])

    adata.obsm[output_key] = MultimodalDiffusionMaps(n_jobs).fit_transform(modalities = [adata.obsm[key] for key in modalities],
                                                                           n_comps = n_components,
                                                                           nn = adata.uns[neighbors_key],
                                                                           epsilons = adata.uns[epsilons_key],
                                                                           weights = np.asarray(adata.obsm[weights_key]),
                                                                           unimodal_norm = unimodal_norm,
                                                                           eigval_times_eigvec = eigval_times_eigvec,
                                                                           random_state = random_state,
                                                                           verbose = verbose)

    if verbose:
        print('{} Multimodal Diffusion Maps components calculated.'.format(n_components))
    
    return adata if copy else None
