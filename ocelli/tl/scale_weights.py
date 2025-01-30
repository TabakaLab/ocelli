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
    """
    Scale multimodal weights for selected cells and modalities

    This function scales the weights of specific cells (`obs`) and modalities by a factor of `kappa`. 
    It is useful for adjusting the influence of certain modalities for selected cells in downstream analyses.

    When selecting observations and modalities, ensure that their types match the 
    `adata.obsm[weights].index` and `adata.obsm[weights].columns`, respectively.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param obs: List of elements from `adata.obsm[weights].index` specifying the cells to scale.
    :type obs: list

    :param modalities: List of elements from `adata.obsm[weights].columns` specifying the modalities to scale.
    :type modalities: list

    :param weights: Key in `adata.obsm` where the weights are stored. (default: `'weights'`)
    :type weights: str

    :param kappa: The scaling factor applied to the selected weights. (default: 1.0)
    :type kappa: float

    :param verbose: Whether to print progress notifications. (default: `False`)
    :type verbose: bool

    :param copy: Whether to return a copy of `adata`. If `False`, updates are made in-place. (default: `False`)
    :type copy: bool

    :returns:
        - If `copy=False`: Updates `adata` with the scaled weights in `adata.obsm[weights]`.
        - If `copy=True`: Returns a modified copy of `adata` with the scaled weights.

    :rtype: anndata.AnnData or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np
            import pandas as pd

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))
            adata.obsm["weights"] = pd.DataFrame(
                np.random.rand(100, 3),
                index=["cell{}".format(i) for i in range(100)],
                columns=["modality1", "modality2", "modality3"]
            )

            # Scale weights for specific cells and modalities
            oci.tl.scale_weights(
                adata,
                obs=["cell1", "cell2"],
                modalities=["modality1", "modality2"],
                kappa=2.0,
                verbose=True
            )
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
