import ocelli as oci
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec


def projections(adata,
                x: str,
                c: str = None,
                phis: list = [10, 55, 100, 145, 190, 235, 280, 225], 
                thetas: list = [10, 55, 100, 145],
                cdict: dict = None,
                markersize: float = 1.,
                markerscale: float = 1.,
                fontsize: float = 6.,
                figsize: tuple = None,
                showlegend: bool = True,
                title: str = None,
                cmap = None,
                vmin: float = None,
                vmax: float = None,
                random_state = None,
                save: str = None,
                dpi: int=300):
    """Plot 3D data from different angles
    
    The function uses ``ocelli.tl.project_2d`` and plots 
    a grid of various perspectives at 3D data.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    x_key
        ``adata.obsm`` key storing a 3D embedding for plotting.
    color_key
        A key of ``adata.obs`` with color scheme. (default: :obj:`None`)
    alphas
        A list of polar coordinates in degrees. (default: [0, 45, 90, 135])
    betas
        A list of polar coordinates in degrees. (default: [0, 45, 90, 135])
    marker_size
        Size of scatter plot markers. (default: 3.)
    fontsize
        Plot fontsize. (default: 6)    
    cmap
        Used only when ``method = matplotlib``. Can be a name (:class:`str`) 
        of a built-in :class:`matplotlib` colormap, 
        or a custom colormap object. (default: ``Spectral``)
    random_state
        Pass an :obj:`int` for reproducible results across multiple function calls. (default: :obj:`None`)

        
    Returns
    -------
    :class:`plotly.graph_objs._figure.Figure`
        A :class:`Plotly` figure if ``static = False``.
    :class:`tuple`
        :class:`matplotlib.figure.Figure` and :class:`numpy.ndarray` 
        storing :class:`matplotlib` figure and axes if ``static = True``.
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
            
            
            if is_discrete:
                groups = np.unique(colors)
                if cdict is None:
                    cdict = {g: plt.get_cmap('jet')(j / (groups.shape[0] - 1)) for j, g in enumerate(groups)}

                ax = fig.add_subplot(gs[i, 2*j])
                ax.set_title('phi={} theta={}'.format(phi, theta) if title is None else title, fontsize=fontsize)
                ax.set_aspect('equal')
                ax.axis('off')
                ax.scatter(x=adata.obsm['X_proj_phi{}_theta{}'.format(phi, theta)][:, 0], 
                           y=adata.obsm['X_proj_phi{}_theta{}'.format(phi, theta)][:, 1],  
                           s=markersize, 
                           c=[cdict[color] for color in colors],
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
                sc = ax.scatter(x=adata.obsm['X_proj_phi{}_theta{}'.format(phi, theta)][:, 0], 
                                y=adata.obsm['X_proj_phi{}_theta{}'.format(phi, theta)][:, 1], 
                                s=markersize, 
                                c=colors,
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
