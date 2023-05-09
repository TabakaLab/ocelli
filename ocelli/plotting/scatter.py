import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec


def scatter(adata: anndata.AnnData,
            x: str,
            markersize: float = 1.,
            c: str = None,
            cdict: dict = None,
            cmap = None,
            vmin = None,
            vmax = None,
            xlim: tuple = None,
            ylim: tuple = None,
            figsize = None,
            ncols: int = 4,
            fontsize: int = 6,
            title: str = None,
            showlegend: bool = True,
            markerscale: float = 1.,
            save: str = None,
            dpi: int = 300):
    """2D scatter plots
    
    Plots a 2D scatter plot using a single (`adata.obs`) or multiple (`adata.obsm`) color schemes.
    The latter case results in a grid of plots.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    x
        `adata.obsm` key with 2D data.
    markersize
         The marker size. (default: 1.)
    c
        `adata.obs` or `adata.obsm` key with a color scheme. (default: :obj:`None`)
    cdict
        Applied when color scheme is discrete. Can be a dictionary mapping color scheme groups to colors. (default: :obj:`None`)
    cmap
        Applied when color scheme is continuous. Can be a name (:class:`str`) 
        of a built-in :class:`matplotlib` colormap, or a custom colormap object. (default: :obj:`None`)
    vmin
        Applied when color scheme is continuous. Lower bound of color scheme. (default: :obj:`None`)
    vmax
        Applied when color scheme is continuous. Upper bound of color scheme. (default: :obj:`None`)
    xlim
        Restrict x-limits of the axis. (default: :obj:`None`)
    ylim
        Restrict y-limits of the axis. (default: :obj:`None`)
    figsize
        Plot figure size. (default: :obj:`None`)
    ncols
        If color scheme is from `adata.obsm`, `ncols` defines the number of columns of plotting grid. (default: 4)
    fontsize
        Plot font size. (default: 6.)
    title
        Plot title. (default: :obj:`None`)
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
        
    df = pd.DataFrame(adata.obsm[x], columns=['x', 'y'])
        
    cnames = []
        
    if c is not None:
        cobs = True if c in list(adata.obs.keys()) else False
        cobsm = True if c in list(adata.obsm.keys()) else False
        
        if cobs and cobsm:
            raise(NameError('Confusion between adata.obs["{}"] and adata.obsm["{}"]. Specify a key unique.'.format(c, c)))
        if not cobs and not cobsm:
             raise(NameError('Wrong parameter c.'))
                
        if cobs:
            nrow = 1
            ncol = 1
            cnames = [c]
            df[c] = list(adata.obs[c])
        
        elif cobsm:
            nplots = adata.obsm[c].shape[1]
            nrow = nplots // ncols if nplots % ncols == 0 else nplots // ncols + 1
            ncol = ncols if nplots >= ncols else nplots
            
            
            if isinstance(adata.obsm[c], pd.DataFrame):
                cnames = list(adata.obsm[c].columns)
                for col in adata.obsm[c].columns:
                    df[col] = list(adata.obsm[c][col])
            else:
                cnames = [i for i in range(adata.obsm[c].shape[1])]
                for i in range(adata.obsm[c].shape[1]):
                    df[i] = list(adata.obsm[c][:, i])
        
    else:
        nrow = 1
        ncol = 1
        cnames = ['color']
        df['color'] = ['Undefined' for _ in range(df.shape[0])]
        
    df = df.sample(frac=1)
        
    ndim = adata.obsm[x].shape[1]
    
    if ndim != 2:
        raise(ValueError('adata.obsm["{}"] must by 2D.'.format(x)))

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    
    gs = GridSpec(nrow,
                  2*ncol, 
                  figure=fig,
                  width_ratios=np.concatenate([[0.95*(1/ncol), 0.05*(1/ncol)] for _ in range(ncol)]), 
                  height_ratios=[1/nrow for _ in range(nrow)])
    
    for i, col in enumerate(cnames):
        
        is_discrete = False
        for el in df[col]:
            if isinstance(el, str):
                is_discrete = True
                break
                
        if is_discrete:
            groups = np.unique(df[col])
            if cdict is None:
                if groups.shape[0] > 1:
                    cdict = {g: plt.get_cmap('jet')(j / (groups.shape[0] - 1)) for j, g in enumerate(groups)}
                else:
                    cdict = {g: '#000000' for j, g in enumerate(groups)}
            
            ax = fig.add_subplot(gs[i//ncol, 2*(i % ncol)])
            ax.set_title(col if title is None else title, fontsize=fontsize)
            ax.set_aspect('equal')
            ax.axis('off')
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.scatter(x=df['x'], y=df['y'], s=markersize, c=[cdict[key] for key in df[col]], edgecolor='none')
            
            if showlegend:
                patches = [Line2D(range(1), 
                                  range(1), 
                                  color="white", 
                                  marker='o', 
                                  markerfacecolor=cdict[key], 
                                  label=key) for key in cdict]
                ax = fig.add_subplot(gs[i//ncol, 2*(i % ncol) + 1])
                ax.axis('off')
                
                ax.legend(handles=patches, fontsize=fontsize, borderpad=0, frameon=False, markerscale=markerscale, loc='center')
        else:
            if cmap is None:
                cmap = plt.get_cmap('jet')
                
            ax = fig.add_subplot(gs[i//ncol, 2*(i % ncol)])
            ax.set_title(col if title is None else title, fontsize=fontsize)
            ax.set_aspect('equal')
            ax.axis('off')
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            sc = ax.scatter(x=df['x'], y=df['y'], s=markersize, c=df[col], cmap=cmap, edgecolor='none',
                            vmin=vmin if vmin is not None else np.min(df[col]), 
                            vmax=vmax if vmax is not None else np.max(df[col]))
            
            if showlegend:
                cbar = fig.colorbar(sc, ax=ax, fraction=0.05)
                cbar.ax.tick_params(labelsize=fontsize, length=0)
                cbar.outline.set_color('white')
               
    if save is not None:
        plt.savefig(save, dpi=dpi, facecolor='white')
        plt.close()
    else:
        plt.close()
        return fig
