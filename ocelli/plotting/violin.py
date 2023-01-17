import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches


def violin(adata: ad.AnnData,
        x: str,
        groups: str,
        c: str,
        cdict = None,
        figsize = None,
        fontsize: int = 6,
        showlegend: bool = True,
        markerscale: float = 1.,
        save: str = None,
        dpi: int = 300):
    """Feature vioolin plots
    
    Generates violin plots showing distributions of feature values taken from `adata.obs` or `adata.obsm`.
    Separate violin plots are created for different groups/clusters of cells as defined by `adata.obs[groups]`.
    
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
    
    groups_unique = np.unique(adata.obs[groups])
    ncol = 2
    
    gs = GridSpec(nrow,
                  ncol, 
                  figure=fig,
                  width_ratios=[0.95*(1/(ncol-1)) for _ in range(ncol-1)] + [0.05], 
                  height_ratios=[1/nrow for _ in range(nrow)])                
            
    
    
    if cdict is None:
        cdict = {g: plt.get_cmap('jet')(j / (groups_unique.shape[0] - 1)) for j, g in enumerate(groups_unique)}
        
    for i, m in enumerate(cnames):
        ax = fig.add_subplot(gs[i, 0])
        ax.set_ylabel(m, fontsize=fontsize)
        ax.set_xticks([])
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=fontsize)
        ax.grid(False)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(0.2)
        ax.tick_params(width=0.2)
        
        data = [np.asarray(adata[adata.obs[groups] == g].obsm[c][m]) for g in groups_unique]

        
            
        violin = ax.violinplot(dataset=data, showextrema=False, showmedians=True)

        for j, pc in enumerate(violin['bodies']):
            pc.set_facecolor(cdict[groups_unique[j]])
            pc.set_edgecolor('black')
            pc.set_linewidth(0.1)
            pc.set_alpha(1)

        vp = violin['cmedians']
        vp.set_edgecolor('black')
        vp.set_linewidth(0.5)
            
        if showlegend:
            ax = fig.add_subplot(gs[:, 1])
            ax.axis('off')
            ax.legend([mpatches.Patch(color=cdict[g]) for g in groups_unique], groups_unique,
                      frameon=False, fontsize=fontsize, loc='upper center', borderpad=0, markerscale=markerscale)    
    if save is not None:
        plt.savefig(save, dpi=dpi, facecolor='white')
        plt.close()
    else:
        plt.close()
        return fig
