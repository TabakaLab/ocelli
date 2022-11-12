import matplotlib.pyplot as plt
import scanpy as sc
import anndata
import matplotlib as mpl
import numpy as np

def weights_per_cluster(adata, 
                        x_key: str = 'X_mdm', 
                        height: str = 'median', 
                        resolution: float = 1., 
                        n_neighbors: int = 20, 
                        clusters_key: str = 'louvain', 
                        weights_key: str = 'weights', 
                        cmap = 'jet', 
                        fontsize: float = 6.,
                        verbose: bool = False,
                        random_state = None):
    """Multimodal weights bar plot
    
    Louvain clustering and plotting median or mean weights per cluster.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    x_key
        ``adata.obsm`` key storing a 2D or 3D embedding for plotting. (default: `X_mdm`)
    height
        Height of bar plots. Valid options: `median`, `mean`. (default: `median`) 
    resolution
        Louvain resolution. (default: 1.)
    n_neighbors
        Louvain nearest neighbors number. (default: 20)
    clusters_key
        Cluster ids will be saved to ``adata.obs[clusters_key]``. (defualt: `louvain`)
    weights_key
        ``adata.obsm[weights_key]`` stores multimodal weights. (default: ``weights``)
    cmap
        Can be a name (:class:`str`) 
        of a built-in :class:`matplotlib` colormap, 
        or a custom colormap object. (default: ``jet``)
    fontsize
        Plot fontsize. (default: 6.)
    verbose
        Print progress notifications. (default: ``False``)
    random_state
        Pass an :obj:`int` for reproducible results across
        multiple function calls. (default: :obj:`None`)
    
    Returns
    -------
    :class:`tuple`
        :class:`matplotlib.figure.Figure` and :class:`numpy.ndarray` 
        storing :class:`matplotlib` figure and axes if ``static = True``.
    """
    
    cmap = mpl.cm.get_cmap(cmap) if type(cmap) == str else cmap
    
    louvain = anndata.AnnData(adata.obsm[x_key])
    sc.pp.neighbors(louvain, n_neighbors=n_neighbors)
    sc.tl.louvain(louvain, resolution=resolution, random_state=random_state)
    
    if verbose:
        print('[{}]\tLouvain clusters computed.'.format(x_key))
    
    adata.obs[clusters_key] = list(louvain.obs['louvain'])
    
    clusters = np.unique([int(cl) for cl in louvain.obs['louvain']])
    modalities = list(adata.obsm[weights_key].columns)
    n_modalities = len(modalities)

    fig, ax = plt.subplots()

    cmap = mpl.cm.get_cmap(cmap) if type(cmap) == str else cmap
    colors = {m: cmap(i/(n_modalities-1)) if n_modalities > 1 else 0.5 for i, m in enumerate(modalities)}

    d = {m: {cl: 0 for cl in clusters} for m in adata.obsm[weights_key].columns}
    for i, cl in enumerate(clusters):
        for j, m in enumerate(modalities):
            if height is 'median':
                h = np.median(adata[adata.obs[clusters_key] == str(cl)].obsm[weights_key][m])
            elif height is 'mean':
                h = np.mean(adata[adata.obs[clusters_key] == str(cl)].obsm[weights_key][m])
            else:
                raise(NameError('Wrong height name. Valid options: median, mean.'))
                
            ax.bar(x=i-(1/3) + (1/6) + j*(2/3)/n_modalities, 
                   height=h, width=(2/3)/n_modalities, label=m if i == 0 else '_', color=colors[m])

    ax.set_xticks(range(clusters.shape[0]), labels=clusters)
    ax.legend(framealpha=0, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    
    ax.set_ylabel('{} weight in cluster'.format(height), fontsize=fontsize)
    ax.set_xlabel('Cluster', fontsize=fontsize)
    
    return fig, ax
