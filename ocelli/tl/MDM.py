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
        save_eigvec: bool = True,
        save_eigval: bool = True,
        save_mmc: bool = False,
        n_jobs: int = -1,
        random_state = None,
        verbose: bool = False,
        copy: bool = False):
    """
    Multimodal Diffusion Maps

    This function computes a multimodal latent space based on weighted modalities using diffusion maps. 
    Each modality contributes to the multimodal Markov chain proportionally to its weights, enabling 
    integrative single-cell analysis across modalities.

    .. note::
        It is necessary to run `ocelli.pp.neighbors` before using this function to ensure 
        that nearest neighbors and distances are computed.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param n_components: Number of MDM components to compute. (default: 10)
    :type n_components: int

    :param modalities: List of keys in `adata.obsm` storing the modalities. If `None`, the list is loaded 
        from `adata.uns['modalities']`. (default: `None`)
    :type modalities: list or None

    :param weights: Key in `adata.obsm` storing modality weights. (default: `'weights'`)
    :type weights: str

    :param out: Key in `adata.obsm` where the MDM embedding is saved. (default: `'X_mdm'`)
    :type out: str

    :param bandwidth_reach: Index of the nearest neighbor used for calculating kernel bandwidths (epsilons). (default: 20)
    :type bandwidth_reach: int

    :param unimodal_norm: Whether to normalize unimodal kernel matrices mid-training. (default: `True`)
    :type unimodal_norm: bool

    :param eigval_times_eigvec: Whether to scale eigenvectors by eigenvalues in the final embedding. 
        If `False`, only eigenvectors are used. (default: `True`)
    :type eigval_times_eigvec: bool

    :param save_eigvec: Whether to save the eigenvectors to `adata.uns['eigenvectors']`. (default: `True`)
    :type save_eigvec: bool

    :param save_eigval: Whether to save the eigenvalues to `adata.uns['eigenvalues']`. (default: `True`)
    :type save_eigval: bool

    :param save_mmc: Whether to save the multimodal Markov chain matrix to `adata.uns['multimodal_markov_chain']`. (default: `False`)
    :type save_mmc: bool

    :param n_jobs: Number of parallel jobs to use. If `-1`, all CPUs are used. (default: `-1`)
    :type n_jobs: int

    :param random_state: Seed for reproducibility. If `None`, no seed is set. (default: `None`)
    :type random_state: int or None

    :param verbose: Whether to print progress notifications. (default: `False`)
    :type verbose: bool

    :param copy: Whether to return a copy of `adata`. If `False`, updates are made in-place. (default: `False`)
    :type copy: bool

    :returns:
        - If `copy=False`: Updates `adata` with the following fields:
            - `adata.obsm[out]`: MDM embedding.
            - `adata.uns['multimodal_markov_chain']` (if `save_mmc=True`): Multimodal Markov chain.
            - `adata.uns['eigenvectors']` (if `save_eigvec=True`): Eigenvectors.
            - `adata.uns['eigenvalues']` (if `save_eigval=True`): Eigenvalues.
        - If `copy=True`: Returns a modified copy of `adata` with these fields.

    :rtype: anndata.AnnData or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))
            adata.obsm['modality1'] = np.random.rand(100, 10)
            adata.obsm['modality2'] = np.random.rand(100, 15)
            adata.uns['modalities'] = ['modality1', 'modality2']

            # Compute nearest neighbors
            oci.pp.neighbors(adata, x=['modality1', 'modality2'], n_neighbors=20)

            # Compute modality weights
            oci.tl.modality_weights(adata, n_jobs=4, verbose=True)

            # Run Multimodal Diffusion Maps
            oci.tl.MDM(adata, n_components=10, verbose=True)
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
