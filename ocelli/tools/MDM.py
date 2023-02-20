import numpy as np
from scipy.sparse import coo_matrix, diags, issparse
from scipy.sparse.linalg import eigsh
from multiprocessing import cpu_count
import ray
import anndata as ad


def adjusted_gaussian_kernel(x, y, epsilon_x, epsilon_y):
    
    epsilons_mul = epsilon_x * epsilon_y
    
    if epsilons_mul == 0:
        return 0
    
    else:
        x = x.toarray().flatten() if issparse(x) else x.flatten()
        y = y.toarray().flatten() if issparse(y) else y.flatten()

        return np.exp(-1 * np.power(np.linalg.norm(x - y), 2) / epsilons_mul)


@ray.remote
def affinity_worker(modality_ref, weights_ref, nn_ref, epsilons_ref, split):
    
    affinities = list()
    for obs0 in split:
        for obs1 in nn_ref[obs0]:
            if obs0 in nn_ref[obs1] and obs0 < obs1:
                pass
            else:
                affinity = adjusted_gaussian_kernel(modality_ref[obs0],
                                                    modality_ref[obs1],
                                                    epsilons_ref[obs0],
                                                    epsilons_ref[obs1])

                mean_weight = (weights_ref[obs0] + weights_ref[obs1]) / 2
                
                affinities.append([obs0, obs1, affinity * mean_weight])
                affinities.append([obs1, obs0, affinity * mean_weight])

    return np.asarray(affinities)


def MDM(adata: ad.AnnData,
        n_components: int = 10,
        modalities: list = None,
        weights: str = 'weights',
        out: str = 'X_mdm',
        bandwidth_reach: int = 20,
        unimodal_norm: bool = True,
        eigval_times_eigvec: bool = True,
        save_eigvec: bool = False,
        save_eigval: bool = False,
        save_mmc: bool = False,
        n_jobs: int = -1,
        random_state = None,
        verbose: bool = False,
        copy: bool = False):
    """Multimodal Diffusion Maps

    This function computes multimodal latent space based on weighted modalities. 

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_components
        Number of MDM components. (default: 10)
    modalities
        List of `adata.obsm` keys storing modalities.
        If :obj:`None`, modalities' keys are loaded from `adata.uns[modalities]`. (default: :obj:`None`)
    weights
        `adata.obsm` key storing modality weights. (default: `weights`)
    out
        `adata.obsm` key where MDM embedding is saved. (default: `X_mdm`)
    bandwidth_reach
        Index of nearest neighbor used for calculating epsilons. (default: 20)
    unimodal_norm
        If `True`, unimodal kernel matrices are normalized mid-training. (default: `True`)
    eigval_times_eigvec
        If `True`, the MDM embedding is calculated by multiplying
        eigenvectors by eigenvalues. Otherwise, the embedding consists of eigenvectors. (default: `True`)
    save_mmc
        If `True`, the multimodal Markov chain array is saved to `adata.uns['multimodal_markov_chain']`. (default: `False`)
    save_eigvec
        If `True`, eigenvectors are saved to `adata.uns['eigenvectors']`. (default: `False`)
    save_eigval
        If `True`, eigenvalues are saved to `adata.uns['eigenvalues']`. (default: `False`)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    random_state
        Pass an :obj:`int` for reproducible results across multiple function calls. (default: :obj:`None`)
    verbose
        Print progress notifications. (default: `False`)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: `False`)

    Returns
    -------
    :obj:`None`
        By default (`copy=False`), updates `adata` with the following fields:
        `adata.obsm[out]` (MDM embedding),
        `adata.uns['mmc']` (if `save_mmc = True`, multimodal Markov chain),
        `adata.uns['eigenvectors']` (if `save_eigvec = True`, eigenvectors),
        `adata.uns['eigenvalues']` (if `save_eigval = True`, eigenvalues).
    :class:`anndata.AnnData`
        When `copy=True` is set, a copy of `adata` with those fields is returned.
    """

    modality_names = adata.uns['modalities'] if modalities is None else modalities
 
    if len(modality_names) == 0:
        raise(NameError('No modality keys found in adata.uns[modalities].'))

    if weights not in adata.obsm:
        raise(KeyError('No weights found in adata.uns["{}"]. Run ocelli.tl.weights.'.format(weights)))
    
    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])
    
    if not ray.is_initialized():
        ray.init(num_cpus=n_jobs)
    
    splits = np.array_split(range(adata.shape[0]), n_jobs)

    for i, m in enumerate(modality_names):
        modality_ref = ray.put(adata.obsm[m])
        weights_ref = ray.put(np.asarray(adata.obsm[weights])[:, i])
        nn_ref = ray.put(adata.obsm['neighbors_{}'.format(m)])
        epsilons_ref = ray.put(adata.obsm['distances_{}'.format(m)][:, bandwidth_reach - 1])
        
        uniM = ray.get([affinity_worker.remote(modality_ref, weights_ref, nn_ref, epsilons_ref, split) for split in splits])
        uniM = np.vstack(uniM)
        uniM = coo_matrix((uniM[:, 2], (uniM[:, 0], uniM[:, 1])), shape=(adata.shape[0], adata.shape[0])).tocsr()

        if unimodal_norm:
            diag_vals = np.asarray([1 / val if val != 0 else 0 for val in uniM.sum(axis=1).A1])
            uniM = diags(diag_vals) @ uniM

        if i == 0:
            multiM = uniM
        else:
            multiM += uniM

        if verbose:
            print('[{}]\tUnimodal Markov chain calculated.'.format(m))
            
    diag_vals = np.asarray([1 / val if val != 0 else 1 for val in multiM.sum(axis=1).A1])
    multiM = diags(diag_vals) @ multiM
            
    multiM = multiM + multiM.T

    if verbose:
        print('Multimodal Markov chain calculated.')
        
    if save_mmc:
        adata.uns['multimodal_markov_chain'] = multiM

    if random_state is not None:
        np.random.seed(random_state)
        v0 = np.random.normal(0, 1, size=(multiM.shape[0]))
    else:
        v0 = None
        
    eigvals, eigvecs = eigsh(multiM, k=n_components + 11, which='LA', maxiter=100000, v0=v0)
    eigvals = np.flip(eigvals)[1:n_components + 1]
    eigvecs = np.flip(eigvecs, axis=1)[:, 1:n_components + 1]
    
    if save_eigvec:
        adata.uns['eigenvectors'] = eigvecs
    if save_eigval:
        adata.uns['eigenvalues'] = eigvals
        
    adata.obsm[out] = eigvecs * eigvals if eigval_times_eigvec else eigvecs

    if verbose:
        print('Eigendecomposition finished.')
        print('{} Multimodal Diffusion Maps components calculated.'.format(n_components))
        
    if ray.is_initialized():
        ray.shutdown()
    
    return adata if copy else None
