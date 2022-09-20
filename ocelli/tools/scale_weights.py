import numpy as np
import anndata
import pandas as pd


def scale_weights(adata: anndata.AnnData,
                  weights_key: str = 'weights',
                  obs: list = [],
                  views: list = [],
                  kappa: float = 1.,
                  verbose: bool = False,
                  copy: bool = False):
    """Multimodal weights scaling
    
    Weights of selected observations (cells) and views are scaled by the factor of ``kappa``.
    If you wish to increase the impact of certain views for some cells, 
    select them and increase ``kappa``.
    
    When selecting views (``views``) and observations (``obs``), pay attention to data types of 
    ``adata.obsm[weights_key].index`` and ``adata.obsm[weights_key].columns``. 
    Your input must match these types.

    Parameters
    ----------
    adata
        The annotated data matrix.
    weights_key
        ``adata.obsm[weights_key]`` stores weights. (default: ``weights``)
    obs
        ``adata.obsm[weights_key].index`` elements storing selected cells. (default: ``[]``)
    views
        ``adata.obsm[weights_key].columns`` elements storing selected views. (default: ``[]``)
    kappa
        The scaling factor. (default: 1)
    verbose
        Print progress notifications. (default: ``False``)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obsm[weights_key]`` (:class:`pandas.DataFrame`).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """
    
    if weights_key not in adata.obsm:
        raise (KeyError('No weights found in adata.uns["{}"]. Run ocelli.tl.weights.'.format(weights_key)))
        
    np_obs_ids = list()
    for i, el in enumerate(adata.obsm[weights_key].index):
        if el in obs:
            np_obs_ids.append(i)
    
    np_view_ids = list()
    for i, el in enumerate(adata.obsm[weights_key].columns):
        if el in views:
            np_view_ids.append(i)
    
    w = np.asarray(adata.obsm[weights_key])
    w[np.ix_(np.unique(np_obs_ids), np.unique(np_view_ids))] *= kappa
    adata.obsm[weights_key] = pd.DataFrame(w, 
                                           index=adata.obsm[weights_key].index, 
                                           columns=adata.obsm[weights_key].columns)
    
    if verbose:
        print('Multimodal cell-specific weights scaled.')
    
    return adata if copy else None
