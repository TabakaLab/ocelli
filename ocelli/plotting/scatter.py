import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib as mpl
from matplotlib.lines import Line2D


def scatter(adata: anndata.AnnData,
            x_key: str,
            color_key = None,
            static: bool = True,
            cmap = 'Spectral',
            marker_size: int = 3):
    """2D and 3D scatter plots
    
    Can generate static 2D plots (:class:`matplotlib`) 
    or interactive 2D and 3D plots (:class:`Plotly`).
    
    Returns :class:`matplotlib` or :class:`Plotly` figures,
    that can be further customized, or saved.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    x_key
        ``adata.obsm`` key storing a 2D or 3D embedding for plotting.
    color_key
        ``adata.obs[color_key]`` stores a discrete or continous information used 
        for coloring the plot. (default: :obj:`None`)
    static
        If ``True``, a plot will be static (available only for 2D). 
        Otherwise, plot will be interactive (2D or 3D). (default: ``True``)
    cmap
        Used only in ``static`` mode. Can be a name (:class:`str`) 
        of a built-in :class:`matplotlib` colormap, 
        or a custom colormap object. (default: ``Spectral``)
    marker_size
        Size of scatter plot markers. (default: 3)
    Returns
    -------
    :class:`plotly.graph_objs._figure.Figure`
        A :class:`Plotly` figure if ``static = False``.
    :class:`tuple`
        :class:`matplotlib.figure.Figure` and :class:`numpy.ndarray` 
        storing :class:`matplotlib` figure and axes if ``static = True``.
    """
    
    if x_key not in list(adata.obsm.keys()):
        raise(NameError('No embedding found to visualize.'))
        
    colors_found = False
    if color_key in list(adata.obs.keys()):
        colors_found = True
        
    dim = adata.obsm[x_key].shape[1]
        
    if static:
        if dim == 2:
            if type(cmap) == str:
                cmap = mpl.cm.get_cmap(cmap)

            df = pd.DataFrame(adata.obsm[x_key], columns=['x', 'y'])
            if colors_found:
                df['color'] = list(adata.obs[color_key])
            else:
                df['color'] = ['Undefined' for _ in range(adata.obsm[x_key].shape[0])]
            df = df.sample(frac=1)
            fig, ax = plt.subplots(1)
            ax.set_aspect('equal')
            try:
                ax.scatter(x=df['x'], y=df['y'], s=marker_size, c=df['color'], cmap=cmap, edgecolor='none')
                scalarmappaple = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=min(df['color']), vmax=max(df['color'])), cmap=cmap)
                scalarmappaple.set_array(256)
                cbar = plt.colorbar(scalarmappaple)
                cbar.ax.tick_params(labelsize=6, length=0)
                cbar.outline.set_color('white')
                plt.axis('off')
            except ValueError:
                types = np.unique(df['color'])
                d = {t: i for i, t in enumerate(types)}
                df['c'] = [d[el] for el in df['color']]
                ax.scatter(x=df['x'], y=df['y'], s=marker_size, c=df['c'], cmap=cmap, edgecolor='none')
                plt.axis('off')
                patches = [Line2D(range(1), range(1), color="white", marker='o', 
                          markerfacecolor=cmap(d[t]/(len(d.keys()))), label=t) for t in d]
                plt.legend(handles=patches, fontsize=4, borderpad=0, frameon=False)
            return fig, ax
        elif dim == 3:
            raise(ValueError('Visualized embedding must be 2-dimensional. You passed {} dimensions. Set static = False.'.format(dim)))
        else:
            raise(ValueError('Visualized embedding must be 2-dimensional. You passed {} dimensions.'.format(dim)))
    else:
        if dim == 2:
            df = pd.DataFrame(adata.obsm[x_key], columns=['x', 'y'])
            if colors_found:
                df['color'] = list(adata.obs[color_key])
            else:
                df['color'] = ['Undefined' for _ in range(adata.obsm[x_key].shape[0])]
            df = df.sample(frac=1)

            fig = px.scatter(df, x='x', y='y', color='color', hover_name='color', 
                        hover_data={'x': False, 'y': False, 'color': False})


            fig.update_layout(scene = dict(
                        xaxis = dict(
                             backgroundcolor='white',
                            visible=False, showticklabels=False,
                             gridcolor="white",
                             showbackground=True,
                             zerolinecolor="white",),
                        yaxis = dict(
                            backgroundcolor='white',
                            visible=False, showticklabels=False,
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="white"),),
                      )
            fig.update_layout({
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white'})
        elif dim == 3:
            df = pd.DataFrame(adata.obsm[x_key], columns=['x', 'y', 'z'])
            if colors_found:
                df['color'] = list(adata.obs[color_key])
            else:
                df['color'] = ['Undefined' for _ in range(adata.obsm[x_key].shape[0])]
            df = df.sample(frac=1)

            fig = px.scatter_3d(df, x='x', y='y', z='z', color='color', hover_name='color', 
                        hover_data={'x': False, 'y': False, 'z': False, 'color': False})

            fig.update_layout(scene = dict(
                xaxis = dict(
                     backgroundcolor='white',
                    visible=False, showticklabels=False,
                     gridcolor="white",
                     showbackground=True,
                     zerolinecolor="white",),
                yaxis = dict(
                    backgroundcolor='white',
                    visible=False, showticklabels=False,
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white"),
                zaxis = dict(
                    backgroundcolor='white',
                    visible=False, showticklabels=False,
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",),),)
            fig.update_layout({
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white'})
        else:
            raise(ValueError('Visualized embedding must be 2- or 3-dimensional. You passed {} dimensions.'.format(dim)))

        fig.update_traces(marker=dict(size=marker_size), selector=dict(mode='markers'))
        fig.update_layout(legend= {'itemsizing': 'constant'})

        return fig