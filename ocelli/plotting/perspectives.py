import ocelli as oci
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def perspectives(adata,
                 x_key,
                 color_key = None,
                 alphas = [0, 45, 90, 135], 
                 betas = [0, 45, 90, 135],
                 marker_size: float = 3.,
                 fontsize: float = 6.,
                 cmap='Spectral',
                 random_state = None):
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

    is_discrete = False
    for el in adata.obs[color_key]:
        if type(el) is str: 
            is_discrete = True
            break
    
    cmap = mpl.cm.get_cmap(cmap) if type(cmap) == str else cmap
    
    colors_unique = np.unique(adata.obs[color_key])
    
    if is_discrete:
        if color_key is None:
            d = {el: cmap(0) for i, el in enumerate(colors_unique)}
        else:
            d = {el: cmap(i/(colors_unique.shape[0]-1)) for i, el in enumerate(colors_unique)}
            
        fig, ax = plt.subplots(len(alphas), len(betas))

        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                oci.tl.project_2d(adata, x_key=x_key, output_key='X_proj', alpha=a, beta=b, random_state=random_state)
                ax[i][j].scatter(x=adata.obsm['X_proj'][:, 0],
                                 y=adata.obsm['X_proj'][:, 1], 
                                 edgecolor='none', 
                                 s=marker_size,
                                 c=[d[col] for col in adata.obs[color_key]])
                ax[i][j].set_aspect('equal')
                ax[i][j].axis('off')
                ax[i][j].set_title('alpha={} beta={}'.format(a, b), fontsize=fontsize)
    else:
        fig, ax = plt.subplots(len(alphas), len(betas))

        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                oci.tl.project_2d(adata, x_key=x_key, output_key='X_proj', alpha=a, beta=b, random_state=random_state)
                ax[i][j].scatter(x=adata.obsm['X_proj'][:, 0],
                                 y=adata.obsm['X_proj'][:, 1], 
                                 edgecolor='none', 
                                 s=marker_size,
                                 c=adata.obs[color_key] if color_key is not None else [0 for _ in range(adata.shape[0])],
                                 cmap=cmap,
                                 vmin=np.min(adata.obs[color_key]) if color_key is not None else 0, 
                                 vmax=np.max(adata.obs[color_key]) if color_key is not None else 0)
                ax[i][j].set_aspect('equal')
                ax[i][j].axis('off')
                ax[i][j].set_title('alpha={} beta={}'.format(a, b), fontsize=fontsize)
                
    return fig, ax
