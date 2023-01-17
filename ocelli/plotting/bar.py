import matplotlib.pyplot as plt
import scanpy as sc
import anndata
import matplotlib as mpl
import numpy as np
from matplotlib.gridspec import GridSpec
import pandas as pd

def bar(adata: anndata.AnnData,
        x: str,
        groups: str,
        c: str,
        height: str = 'median',
        cdict = None,
        figsize = None,
        fontsize: int = 6,
        showlegend: bool = True,
        markerscale: float = 1.,
        save: str = None,
        dpi: int = 300):
    """Feature bar plots
    
    Generates bar plots showing mean or median values of features taken from `adata.obs` or `adata.obsm`.
    Separate bar plots are created for different groups/clusters of cells as defined by `adata.obs[groups]`.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    x
        `adata.obsm` key with 2D data.
    groups
        `adata.obs` key with cell groups.
    c
        `adata.obs` or `adata.obsm` key with a color scheme.
    height
        Height of bars. Valid options: `median`, `mean`. (default: `median`)
    cdict
        A dictionary mapping color scheme groups to colors. (default: :obj:`None`)
    figsize
        Plot figure size. (default: :obj:`None`)
    fontsize
        Plot font size. (default: 6.)
    showlegend
        If `True`, legend is displayed. (default: `True`)
    markerscale
        Changes the size of legend labels. (default: 1.)
    save
        Path for saving the figure. (default: :obj:`None`)
    dpi
        The DPI (Dots Per Inch) of saved image, controls image quality. (default: 300)
    
    Returns
    -------
    :class:`matplotlib.figure.Figure` if `save = None`.
    """

    if x not in list(adata.obsm.keys()):
        raise(NameError('No data found in adata.obsm["{}"].'.format(x)))
        
    if groups not in adata.obs.keys():
        raise(NameError('No data found in adata.obs["{}"].'.format(groups)))
    
    if adata.obsm[x].shape[1] != 2:
        raise(ValueError('adata.obsm["{}"] must by 2D.'.format(x)))
        
    df = pd.DataFrame([], index=[i for i in range(adata.shape[0])])
        
    cnames = []
    
    ncol = 2
        
    cobs = True if c in list(adata.obs.keys()) else False
    cobsm = True if c in list(adata.obsm.keys()) else False

    if cobs and cobsm:
        raise(NameError('Confusion between adata.obs["{}"] and adata.obsm["{}"]. Specify a key unique.'.format(c, c)))
    if not cobs and not cobsm:
         raise(NameError('Wrong parameter c.'))

    if cobs:
        nrow = 1
        cnames = [c]
        df[c] = list(adata.obs[c])

    elif cobsm:
        nrow = adata.obsm[c].shape[1]

        if isinstance(adata.obsm[c], pd.DataFrame):
            cnames = list(adata.obsm[c].columns)
            for col in adata.obsm[c].columns:
                df[col] = list(adata.obsm[c][col])
        else:
            cnames = [i for i in range(adata.obsm[c].shape[1])]
            for i in range(adata.obsm[c].shape[1]):
                df[i] = list(adata.obsm[c][:, i])    

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    
    gs = GridSpec(nrow,
                  ncol, 
                  figure=fig,
                  width_ratios=[0.9, 0.1], 
                  height_ratios=[1/nrow for _ in range(nrow)])                
            
    groups_unique = np.unique(adata.obs[groups])
    
    if cdict is None:
        cdict = {g: plt.get_cmap('jet')(j / (groups_unique.shape[0] - 1)) for j, g in enumerate(groups_unique)}
    
    for j, m in enumerate(cnames):
        ax = fig.add_subplot(gs[j, 0])
        ax.set_ylabel(m, fontsize=fontsize)
        ax.set_xticks([])
        ax.spines[['right', 'top']].set_visible(False)
        
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(0.2)
            
        ax.tick_params(width=0.2)
        ax.tick_params(axis='y', which='major', labelsize=fontsize)
        
        for i, g in enumerate(groups_unique):
            if height is 'median':
                h = np.median(adata[adata.obs[groups] == g].obsm[c][m])
            elif height is 'mean':
                h = np.mean(adata[adata.obs[groups] == g].obsm[c][m])
            else:
                raise(NameError('Wrong height parameter. Valid options: median, mean.'))
            
            ax.bar(x=i, height=h, width=0.6, label=g, color=cdict[g])
            
        
        handles, labels = ax.get_legend_handles_labels()
            
        if showlegend:
            ax = fig.add_subplot(gs[:, 1])
            ax.axis('off')
            ax.legend(handles, labels, frameon=False, fontsize=fontsize, loc='upper center', borderpad=0, markerscale=markerscale)
        
    if save is not None:
        plt.savefig(save, dpi=dpi, facecolor='white')
        plt.close()
    else:
        plt.close()
        return fig
