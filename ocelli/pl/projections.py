import ocelli as oci
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec


def projections(adata,
                x: str,
                s: float = 1.,
                c: str = None,
                phis: list = [10, 55, 100, 145, 190, 235, 280, 225], 
                thetas: list = [10, 55, 100, 145],
                cdict: dict = None,
                cmap = None,
                vmin: float = None,
                vmax: float = None,
                figsize: tuple = None,
                fontsize: float = 6.,
                title: str = None,
                showlegend: bool = True,
                markerscale: float = 1.,
                random_state = None,
                save: str = None,
                dpi: int=300):
    """
    Project and visualize 3D data from multiple angles

    This function creates 2D projections of 3D data stored in `adata.obsm` by projecting it onto 
    planes defined by polar coordinates (`phi`, `theta`) using ``ocelli.tl.projection``. 
    The resulting 2D plots are organized in a grid based on the specified angles.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param x: Key in `adata.obsm` containing the 3D data to project.
    :type x: str

    :param s: Size of the scatter plot markers. (default: 1.0)
    :type s: float

    :param c: Key in `adata.obs` specifying the color scheme. (default: `None`)
    :type c: str or None

    :param phis: List of azimuthal angles (`phi`) in degrees, ranging from 0 to 360. (default: `[10, 55, 100, 145, 190, 235, 280, 225]`)
    :type phis: list

    :param thetas: List of polar angles (`theta`) in degrees, ranging from 0 to 180. (default: `[10, 55, 100, 145]`)
    :type thetas: list

    :param cdict: Dictionary mapping discrete color groups to colors. (default: `None`)
    :type cdict: dict or None

    :param cmap: Colormap for continuous color schemes. (default: `None`)
    :type cmap: str or None

    :param vmin: Lower bound for continuous color schemes. (default: `None`)
    :type vmin: float or None

    :param vmax: Upper bound for continuous color schemes. (default: `None`)
    :type vmax: float or None

    :param figsize: Size of the figure. (default: `None`)
    :type figsize: tuple or None

    :param fontsize: Font size for titles and labels. (default: 6.0)
    :type fontsize: float

    :param title: Title for the subplots. If `None`, defaults to `'phi=..., theta=...'`. (default: `None`)
    :type title: str or None

    :param showlegend: Whether to display a legend for discrete color schemes. (default: `True`)
    :type showlegend: bool

    :param markerscale: Scale for the legend markers. (default: 1.0)
    :type markerscale: float

    :param random_state: Seed for reproducibility of random projections. (default: `None`)
    :type random_state: int or None

    :param save: Path to save the figure as an image. If `None`, the plot is returned as a Matplotlib figure. (default: `None`)
    :type save: str or None

    :param dpi: Resolution of the saved figure in dots per inch (DPI). Controls the image quality. (default: 300)
    :type dpi: int

    :returns:
        - If `save=None`: Returns a Matplotlib figure object.
        - Otherwise, saves the plot to the specified path and does not return anything.
    :rtype: matplotlib.figure.Figure or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))
            adata.obsm["3D_data"] = np.random.rand(100, 3)
            adata.obs["group"] = np.random.choice(["A", "B", "C"], 100)

            # Generate projections
            oci.pl.projections(
                adata,
                x="3D_data",
                c="group",
                phis=[0, 90, 180],
                thetas=[30, 60],
                figsize=(12, 8),
                s=3
            )
    """
    
    if x not in list(adata.obsm.keys()):
        raise(NameError('No data found in adata.obsm["{}"].'.format(x)))
        
    ndim = adata.obsm[x].shape[1]
    
    if ndim != 3:
        raise(ValueError('Projected data must be 3D'))
        
    if c is not None:
        if c not in list(adata.obs.keys()):
            raise(NameError('No data found in adata.obs["{}"].'.format(c)))
    
    colors = list(adata.obs[c]) if c is not None else ['Undefined' for _ in range(adata.shape[0])]
    
    is_discrete = False
    for el in colors:
        if isinstance(el, str):
            is_discrete = True
            break    
    
    nrow = len(thetas)
    ncol = len(phis)
    
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    
    gs = GridSpec(nrow,
                  2*ncol, 
                  figure=fig,
                  width_ratios=np.concatenate([[0.95*(1/ncol), 0.05*(1/ncol)] for _ in range(ncol)]), 
                  height_ratios=[1/nrow for _ in range(nrow)])
    
    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            oci.tl.projection(adata,
                              x=x, 
                              out='X_proj_phi{}_theta{}'.format(phi, theta), 
                              phi=phi, 
                              theta=theta, 
                              random_state=random_state)
            
            df = pd.DataFrame(adata.obsm['X_proj_phi{}_theta{}'.format(phi, theta)], columns=['x', 'y'])
            df['c'] = colors
            df = df.sample(frac=1)
            
            if is_discrete:
                groups = np.unique(colors)
                if cdict is None:
                    if groups.shape[0] > 1:
                        cdict = {g: plt.get_cmap('jet')(j / (groups.shape[0] - 1)) for j, g in enumerate(groups)}
                    else:
                        cdict = {g: '#000000' for g in groups}

                ax = fig.add_subplot(gs[i, 2*j])
                ax.set_title('phi={} theta={}'.format(phi, theta) if title is None else title, fontsize=fontsize)
                ax.set_aspect('equal')
                ax.axis('off')
                ax.scatter(x=df['x'], 
                           y=df['y'],  
                           s=s, 
                           c=[cdict[color] for color in df['c']],
                           edgecolor='none')

                if showlegend:
                    patches = [Line2D(range(1), 
                                      range(1), 
                                      color="white", 
                                      marker='o', 
                                      markerfacecolor=cdict[key], 
                                      label=key) for key in cdict]
                    ax = fig.add_subplot(gs[i, 2*j + 1])
                    ax.axis('off')

                    ax.legend(handles=patches, fontsize=fontsize, borderpad=0, frameon=False, markerscale=markerscale)
            else:
                if cmap is None:
                    cmap = plt.get_cmap('jet')

                ax = fig.add_subplot(gs[i, 2*j])
                ax.set_title('phi={} theta={}'.format(phi, theta) if title is None else title, fontsize=fontsize)
                ax.set_aspect('equal')
                ax.axis('off')
                sc = ax.scatter(x=df['x'], 
                                y=df['y'], 
                                s=s, 
                                c=df['c'],
                                cmap=cmap, 
                                edgecolor='none',
                                vmin=vmin if vmin is not None else np.min(colors), 
                                vmax=vmax if vmax is not None else np.max(colors))

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
