import matplotlib.pyplot as plt
import scanpy as sc
import anndata
import matplotlib as mpl
import numpy as np
from matplotlib.gridspec import GridSpec
import pandas as pd


def bar(adata: anndata.AnnData,
        groups: str,
        values: str,
        height: str = 'median',
        cdict = None,
        figsize = None,
        fontsize: int = 6,
        showlegend: bool = True,
        markerscale: float = 1.,
        save: str = None,
        dpi: int = 300):
    """
    Feature bar plots

    Generates bar plots showing the mean or median values of features from `adata.obs` or `adata.obsm`. 
    Separate bar plots are created for each group or cluster defined in `adata.obs[groups]`.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param groups: Key in `adata.obs` specifying the groups or clusters for which bar plots are generated.
    :type groups: str

    :param values: Key in `adata.obs` or `adata.obsm` specifying the values to be plotted.
    :type values: str

    :param height: The metric used for bar heights. Valid options: `median`, `mean`. (default: `median`)
    :type height: str

    :param cdict: A dictionary mapping groups to colors. If `None`, colors are generated automatically. (default: `None`)
    :type cdict: dict or None

    :param figsize: Size of the plot figure. (default: `None`)
    :type figsize: tuple or None

    :param fontsize: Font size for plot labels and legends. (default: 6)
    :type fontsize: int

    :param showlegend: Whether to display a legend for the groups. (default: `True`)
    :type showlegend: bool

    :param markerscale: Scale of the legend markers. (default: 1.0)
    :type markerscale: float

    :param save: Path to save the plot as an image. If `None`, the plot is returned as a Matplotlib figure. (default: `None`)
    :type save: str or None

    :param dpi: Resolution of the saved figure in dots per inch (DPI). Controls the image quality. (default: 300)
    :type dpi: int

    :returns:
        - If `save=None`: Returns a Matplotlib figure object.
        - Otherwise, saves the plot to the specified path and does not return anything.
    :rtype: matplotlib.figure.Figure or None

    :example:

        .. code-block:: python

            import ocelli.pl as pl
            from anndata import AnnData
            import numpy as np
            import pandas as pd

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))
            adata.obs["groups"] = np.random.choice(["A", "B", "C"], 100)
            adata.obs["values"] = np.random.rand(100)

            # Generate bar plot
            pl.bar(adata, groups="groups", values="values", height="mean", fontsize=10, showlegend=True)
    """

    if groups not in adata.obs.keys():
        raise(NameError('No data found in adata.obs["{}"].'.format(groups)))
        
    df = pd.DataFrame([], index=[i for i in range(adata.shape[0])])
        
    cnames = []
        
    cobs = True if values in list(adata.obs.keys()) else False
    cobsm = True if values in list(adata.obsm.keys()) else False

    if cobs and cobsm:
        raise(NameError('Confusion between adata.obs["{}"] and adata.obsm["{}"]. Specify a key unique.'.format(c, c)))
    if not cobs and not cobsm:
         raise(NameError('Wrong parameter c.'))

    if cobs:
        nrow = 1
        cnames = [values]
        df[values] = list(adata.obs[values])

    elif cobsm:
        nrow = adata.obsm[values].shape[1]

        if isinstance(adata.obsm[values], pd.DataFrame):
            cnames = list(adata.obsm[values].columns)
            for col in adata.obsm[values].columns:
                df[col] = list(adata.obsm[values][col])
        else:
            cnames = [i for i in range(adata.obsm[values].shape[1])]
            for i in range(adata.obsm[values].shape[1]):
                df[i] = list(adata.obsm[values][:, i])

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    
    gs = GridSpec(nrow,
                  2, 
                  figure=fig,
                  width_ratios=[0.95, 0.05], 
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
            if height == 'median':
                h = np.median(adata[adata.obs[groups] == g].obsm[values][m])
            elif height == 'mean':
                h = np.mean(adata[adata.obs[groups] == g].obsm[values][m])
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
