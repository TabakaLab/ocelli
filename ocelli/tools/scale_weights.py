import numpy as np
import anndata
import pandas as pd


def scale_weights(adata: anndata.AnnData,
                  obs: list,
                  modalities: list,
                  weights: str = 'weights',
                  kappa: float = 1.,
                  verbose: bool = False,
                  copy: bool = False):
    """Multimodal weights scaling
    
    Scales weights of selected observations (cells) and modalities by the factor of `kappa`.
    If you wish to increase the impact of certain modalities for some cells, 
    select them and increase `kappa`.
    
    When selecting modalities and observations, pay attention to data types of 
    `adata.obsm[weights].index` and `adata.obsm[weights].columns`. 
    Your input must match these types.

    Parameters
    ----------
    adata
        The annotated data matrix.
    obs
        `adata.obsm[weights].index` elements storing selected observations.
    modalities
        `adata.obsm[weights].columns` elements storing selected modalities.
    weights
        `adata.obsm[weights]` stores weights. (default: `weights`)
    kappa
        The scaling factor. (default: 1)
    verbose
        Print progress notifications. (default: `False`)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: `False`)

    Returns
    -------
    :obj:`None`
        By default (`copy=False`), updates `adata` with the following fields:
        `adata.obsm[weights]` (scaled weights).
    :class:`anndata.AnnData`
        When `copy=True` is set, a copy of `adata` with those fields is returned.
    """
    
    if weights not in adata.obsm:
        raise (KeyError('No weights found in adata.uns["{}"].'.format(weights)))
        
    np_obs_ids = list()
    for i, el in enumerate(adata.obsm[weights].index):
        if el in obs:
            np_obs_ids.append(i)
    
    np_modality_ids = list()
    for i, el in enumerate(adata.obsm[weights].columns):
        if el in modalities:
            np_modality_ids.append(i)
    
    w = np.asarray(adata.obsm[weights])
    w[np.ix_(np.unique(np_obs_ids), np.unique(np_modality_ids))] *= kappa
    adata.obsm[weights] = pd.DataFrame(w, index=list(adata.obsm[weights].index), columns=list(adata.obsm[weights].columns))
    
    if verbose:
        print('Multimodal weights scaled.')
    
    return adata if copy else None
