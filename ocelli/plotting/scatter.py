import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

"""2D and 3D scatter plots
    
    Static :class:`matplotlib` 2D plots,
    or interactive :class:`Plotly` 2D or 3D plots.
    
    Returns :class:`matplotlib` or :class:`Plotly` figures,
    that can be further customized, or saved.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    x_key
        `adata.obsm` key storing a 2D or 3D embedding for plotting.
    color_key
        A key of `adata.obs` or `adata.obsm` with plot coloring information.
        If `method=plotly`, only `adata.obs` keys are valid.
        (default: :obj:`None`)
    method
        Valid options: `matplotlib`, `plotly`.
        `matplotlib` generates static 2D plots.
        `plotly` generates 2D or 3D interactive plots. (default: `None`)
    cmap
        If `method=matplotlib`, `cmap` can be a name (:class:`str`) 
        of a built-in :class:`matplotlib` colormap, or a custom colormap object.
        If `method=plotly`, `cmap` is the value of `color_continuous_scale` 
        parameter of `plotly.express.scatter` or `plotly.express.scatter_3d`. (default: `None`)
    fontsize
        Applicable if `method=matplotlib`. Plot fontsize. (default: 6)
    max_columns
        Applicable if `method=matplotlib` and `color_key` is a `adata.obsm` key. 
        A maximum number of columns for a plot. Must be greater than 2. (default: 4)
    marker_size
        Size of scatter plot markers. (default: 3.)
    markerscale
        Applicable if `method=matplotlib`. Scales marker size in a discrete legend. (default: 1.)
    vmin
        Lower bound of legend colorbar. If `method=plotly`, you must also specify `vmax` value. (default: ``None``)
    vmax
        Upper bound of legend colorbar. If `method=plotly`, you must also specify `vmin` value. (default: ``None``)
    axes_visible
        Make axes visible. (default: ``False``)
    legend
        Applicable if `method=matplotlib`. If ``True``, show legend. (default: ``True``)
        
    Returns
    -------
    :class:`plotly.graph_objs._figure.Figure`
        A :class:`Plotly` figure if ``static = False``.
    :class:`tuple`
        :class:`matplotlib.figure.Figure` and :class:`numpy.ndarray` 
        storing :class:`matplotlib` figure and axes if ``static = True``.
    """


def scatter(adata: anndata.AnnData,
            x: str,
            c = None,
            cdict = None,
            cmap = None,
            vmin = None,
            vmax = None,
            markersize: float = 1.,
            markerscale: float = 1.,
            fontsize: int = 6,
            figsize = None,
            ncols: int = 4,
            showlegend: bool = True,
            title: str = None,
            save: str = None,
            dpi: int = 300):

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
            
            ncol = ncols if nplots >= ncols else nplots % ncols
            
            
            if isinstance(adata.obsm[c], pd.DataFrame):
                cnames = list(adata.obsm[c].columns)
                for col in adata.obsm[c].columns:
                    df[col] = list(adata.obs[c][col])
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
                cdict = {g: plt.get_cmap('jet')(j / (groups.shape[0] - 1)) for j, g in enumerate(groups)}
            
            ax = fig.add_subplot(gs[i//ncol, 2*(i % ncol)])
            ax.set_title(col if title is None else title, fontsize=fontsize)
            ax.set_aspect('equal')
            ax.axis('off')
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
                
                ax.legend(handles=patches, fontsize=fontsize, borderpad=0, frameon=False, markerscale=markerscale)
        else:
            if cmap is None:
                cmap = plt.get_cmap('jet')
                
            ax = fig.add_subplot(gs[i//ncol, 2*(i % ncol)])
            ax.set_title(col if title is None else title, fontsize=fontsize)
            ax.set_aspect('equal')
            ax.axis('off')
            sc = ax.scatter(x=df['x'], y=df['y'], s=markersize, c=df[col], cmap=cmap, edgecolor='none',
                            vmin=vmin if vmin is not None else np.min(df[col]), 
                            vmax=vmax if vmax is not None else np.max(df[col]))
            
            if showlegend:
                cbar = fig.colorbar(sc, ax=ax, fraction=0.1)
                cbar.ax.tick_params(labelsize=fontsize, length=0)
                cbar.outline.set_color('white')
               
    if save is not None:
        plt.savefig(save, dpi=dpi, facecolor='white')
        plt.close()
    else:
        plt.close()
        return fig
