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
            s: float = 1.,
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
    """
    2D scatter plots

    This function creates a 2D scatter plot based on data stored in `adata.obsm` and optionally colors 
    the points using a single (`adata.obs`) or multiple (`adata.obsm`) color schemes. 
    For multiple color schemes, a grid of plots is generated.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param x: Key in `adata.obsm` containing the 2D data for plotting.
    :type x: str

    :param s: Size of the scatter plot markers. (default: 1.0)
    :type s: float

    :param c: Key in `adata.obs` or `adata.obsm` specifying the color scheme. (default: `None`)
    :type c: str or None

    :param cdict: A dictionary mapping discrete groups in the color scheme to specific colors. Used for discrete color schemes. (default: `None`)
    :type cdict: dict or None

    :param cmap: Colormap for continuous color schemes. Can be a string name of a Matplotlib colormap or a custom colormap. (default: `None`)
    :type cmap: str or None

    :param vmin: Lower bound of the colormap for continuous color schemes. (default: `None`)
    :type vmin: float or None

    :param vmax: Upper bound of the colormap for continuous color schemes. (default: `None`)
    :type vmax: float or None

    :param xlim: Tuple defining the x-axis limits. (default: `None`)
    :type xlim: tuple or None

    :param ylim: Tuple defining the y-axis limits. (default: `None`)
    :type ylim: tuple or None

    :param figsize: Tuple specifying the figure size. (default: `None`)
    :type figsize: tuple or None

    :param ncols: Number of columns in the grid when plotting multiple color schemes from `adata.obsm`. (default: 4)
    :type ncols: int

    :param fontsize: Font size for plot labels and titles. (default: 6)
    :type fontsize: int

    :param title: Title of the plot. If multiple plots are generated, this applies to each subplot. (default: `None`)
    :type title: str or None

    :param showlegend: Whether to display a legend for discrete color schemes. (default: `True`)
    :type showlegend: bool

    :param markerscale: Scale of the legend markers. (default: 1.0)
    :type markerscale: float

    :param save: Path to save the plot as a file. If `None`, the plot is returned as a Matplotlib figure. (default: `None`)
    :type save: str or None

    :param dpi: Resolution of the saved figure in dots per inch (DPI). (default: 300)
    :type dpi: int

    :returns: 
        - If `save` is `None`, returns a Matplotlib figure object.
        - Otherwise, saves the plot to the specified path and does not return anything.
    :rtype: matplotlib.figure.Figure or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))
            adata.obsm["embedding"] = np.random.rand(100, 2)
            adata.obs["group"] = np.random.choice(["A", "B", "C"], 100)

            # Generate scatter plot
            oci.pl.scatter(adata, x="embedding", c="group", s=5, title="Scatter Plot Example")
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
            ax.scatter(x=df['x'], y=df['y'], s=s, c=[cdict[key] for key in df[col]], edgecolor='none')
            
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
            sc = ax.scatter(x=df['x'], y=df['y'], s=s, c=df[col], cmap=cmap, edgecolor='none',
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
