import anndata
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def topics(adata: anndata.AnnData,
           x_key: str = 'X_fa2',
           topics_key: str = 'lda',
           cmap = None,
           marker_size: int = 1):
    """Topics scatter plots
    
    Generates scatter plots with topic scores.
    
    Returns a :class:`tuple` of :class:`matplotlib` figure and axes.
    They can be further customized, or saved.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    x_key
        ``adata.obsm[x_key]`` stores a 2D embedding. (default: `X_fa2`)
    topics_key
        ``adata.obsm[topics_key]`` stores topic scores as a matrix of shape ``(n_cells, n_topics)``. (default: ``lda``)
    cmap
        If None, a predfined custom :class:`matplotlib` colormap is used.
        Otherwise, can be a name (:class:`str`) of a built-in :class:`matplotlib` colormap, 
        or a custom colormap object. (default: :obj:`None`)
    marker_size
        Size of scatter plot markers. (default: 1)

    Returns
    -------
    :class:`tuple`
        :class:`matplotlib.figure.Figure` and :class:`numpy.ndarray` storing :class:`matplotlib` figure and axes.
    """
    
    if x_key not in list(adata.obsm.keys()):
        raise(NameError('No embedding found to visualize.'))
        
    if topics_key not in list(adata.obsm.keys()):
        raise(NameError('No topic modeling results found.'))
    
    n_topics = adata.obsm[topics_key].shape[1]
    
    if cmap is None:
        cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', ['#000000', '#0000B6', 
                                                                       '#0000DB', '#0000FF', 
                                                                       '#0055FF', '#00AAFF',
                                                                       '#00FFFF', '#20FFDF', 
                                                                       '#40FFBF', '#60FF9F',
                                                                       '#80FF80', '#9FFF60',
                                                                       '#BFFF40', '#DFFF20',
                                                                       '#FFFF00', '#FFAA00', 
                                                                       '#FF5500', '#FF0000',
                                                                       '#DB0000',  '#B60000'], N=256)
    else:
        if type(cmap) == str:
            cmap = mpl.cm.get_cmap(cmap)
            
    n_topics = adata.obsm[topics_key].shape[1]
    
    n_rows, n_columns = n_topics // 5, n_topics % 5
    if n_columns > 0:
        n_rows += 1
    if n_rows != 1:
        n_columns = 5

    fig, ax = plt.subplots(n_rows, n_columns)
    vmin, vmax = 0, np.percentile(adata.obsm[topics_key].flatten(), 95)
    
    for i in range(n_rows * n_columns):
        df = pd.DataFrame(adata.obsm[x_key], columns=['x', 'y'])
        if i < n_topics:
            im = ax[i // 5][i % 5].scatter(df['x'], df['y'], c=adata.obsm[topics_key][:, i], cmap=cmap, alpha=1, s=marker_size, edgecolors='none', vmin=vmin, vmax=vmax)
            ax[i // 5][i % 5].set_title('{}'.format(i), fontsize=4)
        ax[i // 5][i % 5].axis('off')
        ax[i // 5][i % 5].set_aspect('equal')
        
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    return fig, ax
