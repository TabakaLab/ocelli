from anndata import AnnData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib as mpl
from matplotlib.lines import Line2D


def weights(adata: AnnData,
            weights_key: str = 'weights',
            celltype_key: str = 'celltype',
            showmeans: bool = False, 
            showmedians: bool = True, 
            showextrema: bool = False):
    """Multi-view weights violin plots
    
    Basic violin plots of multi-view weights. 
    A seperate violin plot is generated for each view and celltype. 
    Looks best when the numbers of views and cell types are not large.
    
    Returns a :class:`tuple` of :class:`matplotlib` figure and axes.
    They can be further customized, or saved.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    weights_key
        ``adata.obsm[weights_key]`` stores multi-view weights. (default: ``weights``)
    celltype_key
        ``adata.obs[celltype_key]`` stores celltypes. For each celltype 
        a seperate violin plot is generated. If ``celltype_key`` is not found, 
        violin plots for all cells are generated. (default: ``celltype``)
    showmeans
        If ``True``, will toggle rendering of the means. (default: ``False``)
    showmedians
        If ``True``, will toggle rendering of the medians. (default: ``True``)
    showextrema
        If ``True``, will toggle rendering of the extrema. (default: ``False``)

    Returns
    -------
    :class:`tuple`
        :class:`matplotlib.figure.Figure` and :class:`numpy.ndarray` storing :class:`matplotlib` figure and axes.
    """
    
    if celltype_key not in list(adata.obs.keys()):
        views = list(adata.obsm['weights'].columns)
        fig, ax = plt.subplots(nrows=len(views), ncols=1)
        fig.supylabel('views', size=6)
        fig.suptitle('weights', size=6)

        for i, view in enumerate(views):
            ax[i].violinplot(adata.obsm['weights'][view], 
                             showmeans=showmeans, 
                             showmedians=showmedians, 
                             showextrema=showextrema)
            ax[i].set_ylabel(view, size=6)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['bottom'].set_visible(False)
            ax[i].set_xticks([])
            ax[i].tick_params(axis='both', which='major', labelsize=6, length=1)
            ax[i].set_yticks([0, 0.5, 1])
            ax[i].set_yticklabels([0, 0.5, 1])
            ax[i].set_ylim([0,1])
        plt.tight_layout()
    else:
        views = list(adata.obsm['weights'].columns)
        celltypes = list(np.unique(adata.obs[celltype_key]))

        fig, ax = plt.subplots(nrows=len(views), ncols=len(celltypes))
        fig.supylabel('views', size=6)
        fig.suptitle('celltypes', size=6)

        for i, view in enumerate(views):
            for j, celltype in enumerate(celltypes):
                ax[i][j].violinplot(adata[adata.obs[celltype_key] == celltype].obsm[weights_key][view], 
                                    showmeans=showmeans, 
                                    showmedians=showmedians, 
                                    showextrema=showextrema)
                if i == 0:
                    ax[i][j].set_title(celltypes[j], size=6)
                if j == 0:
                    ax[i][j].set_ylabel(view, size=6)
                ax[i][j].spines['right'].set_visible(False)
                ax[i][j].spines['top'].set_visible(False)
                ax[i][j].spines['bottom'].set_visible(False)
                ax[i][j].set_xticks([])
                ax[i][j].tick_params(axis='both', which='major', labelsize=6, length=1)
                ax[i][j].set_yticks([0, 0.5, 1])
                if j == 0:
                    ax[i][j].set_yticklabels([0, 0.5, 1])
                else:
                    ax[i][j].set_yticklabels([])
                ax[i][j].set_ylim([0,1])
        plt.tight_layout()
    
    return fig, ax


def scatter(adata: AnnData,
            static: bool = True,
            x_key: str = 'x_fa2',
            color_key: str = 'celltype',
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
    static
        If ``True``, a plot will be static (available only for 2D). 
        Otherwise, plot will be interactive (2D or 3D). (default: ``True``)
    x_key
        ``adata.obsm[x_key]`` stores a 2D or 3D embedding. (default: ``x_fa2``)
    color_key
        ``adata.obs[color_key]`` stores a discrete or continous information used 
        for coloring the plot. (default: ``celltype``)
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
        
    colors = True
    if color_key not in list(adata.obs.keys()):
        print('No colors found. Plot will not contain colors.')
        colors = False
        
    dim = adata.obsm[x_key].shape[1]
        
    if static:
        if dim == 2:
            if type(cmap) == str:
                cmap = mpl.cm.get_cmap(cmap)

            df = pd.DataFrame(adata.obsm[x_key], columns=['x', 'y'])
            df[color_key] = list(adata.obs[color_key])
            df = df.sample(frac=1)
            fig, ax = plt.subplots(1)
            ax.set_aspect('equal')
            try:
                ax.scatter(x=df['x'], y=df['y'], s=marker_size, c=df[color_key], cmap=cmap)
                scalarmappaple = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=min(df[color_key]), vmax=max(df[color_key])), cmap=cmap)
                scalarmappaple.set_array(256)
                cbar = plt.colorbar(scalarmappaple)
                cbar.ax.tick_params(labelsize=6, length=0)
                cbar.outline.set_color('white')
                plt.axis('off')
            except ValueError:
                types = np.unique(df[color_key])
                d = {t: i for i, t in enumerate(types)}
                df['c'] = [d[el] for el in df[color_key]]
                ax.scatter(x=df['x'], y=df['y'], s=marker_size, c=df['c'], cmap=cmap)
                plt.axis('off')
                patches = [Line2D(range(1), range(1), color="white", marker='o', 
                          markerfacecolor=cmap(d[t]/(len(d.keys())-1)), label=t) for t in d]
                plt.legend(handles=patches, fontsize=6, borderpad=0, frameon=False)
            return fig, ax
        elif dim == 3:
            raise(ValueError('Visualized embedding must be 2-dimensional. You passed {} dimensions. Set static = False.'.format(dim)))
        else:
            raise(ValueError('Visualized embedding must be 2-dimensional. You passed {} dimensions.'.format(dim)))
    else:
        if dim == 2:
            df = pd.DataFrame(adata.obsm[x_key], columns=['x', 'y'])
            df[color_key] = list(adata.obs[color_key])
            df = df.sample(frac=1)

            fig = px.scatter(df, x='x', y='y', color=color_key, hover_name=color_key, 
                        hover_data={'x': False, 'y': False, color_key: False})


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
            df[color_key] = list(adata.obs[color_key])
            df = df.sample(frac=1)

            fig = px.scatter_3d(df, x='x', y='y', z='z', color=color_key, hover_name=color_key, 
                        hover_data={'x': False, 'y': False, 'z': False, color_key: False})

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

    
def topics(adata: AnnData,
           x_key: str = 'x_fa2',
           topics_key: str = 'lda',
           cmap = None,
           marker_size: int = 1):
    """Topics scatter plots
    
    Generates scatter plots with topic scores.
    
    Returns a :class:`tuple` of :class:`matplotlib` figure and axes.
    They can be further customized, or saved.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    x_key
        ``adata.obsm[x_key]`` stores a 2D embedding. (default: ``x_fa2``)
    topics_key
        ``adata.obsm[topics_key]`` stores topic scores as a matrix of shape ``(n_cells, n_topics)``. (default: ``lda``)
    cmap
        If None, a predfined custom :class:`matplotlib` colormap is used.
        Otherwise, can be a name (:class:`str`) of a built-in :class:`matplotlib` colormap, 
        or a custom colormap object. (default: :obj:`None`)
    marker_size
        Size of scatter plot markers. (default: 1)

    Returns
    -------
    :class:`tuple`
        :class:`matplotlib.figure.Figure` and :class:`numpy.ndarray` storing :class:`matplotlib` figure and axes.
    """
    
    if x_key not in list(adata.obsm.keys()):
        raise(NameError('No embedding found to visualize.'))
        
    if topics_key not in list(adata.obsm.keys()):
        raise(NameError('No topic modeling results found.'))
    
    n_topics = adata.obsm[topics_key].shape[1]
    
    if cmap is None:
        cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', ['#000000', '#0000B6', 
                                                                       '#0000DB', '#0000FF', 
                                                                       '#0055FF', '#00AAFF',
                                                                       '#00FFFF', '#20FFDF', 
                                                                       '#40FFBF', '#60FF9F',
                                                                       '#80FF80', '#9FFF60',
                                                                       '#BFFF40', '#DFFF20',
                                                                       '#FFFF00', '#FFAA00', 
                                                                       '#FF5500', '#FF0000',
                                                                       '#DB0000',  '#B60000'], N=256)
    else:
        if type(cmap) == str:
            cmap = mpl.cm.get_cmap(cmap)
            
    n_topics = adata.obsm[topics_key].shape[1]
    
    n_rows, n_columns = n_topics // 5, n_topics % 5
    if n_columns > 0:
        n_rows += 1
    if n_rows != 1:
        n_columns = 5

    fig, ax = plt.subplots(n_rows, n_columns)
    for i in range(n_rows * n_columns):
        if i < n_topics:
            df = pd.DataFrame(adata.obsm[x_key], columns=['x', 'y'])

            ax[i // 5][i % 5].scatter(df['x'], df['y'], c=adata.obsm[topics_key][:, i], 
                                      cmap=cmap, alpha=1, s=marker_size, edgecolors='none')
            ax[i // 5][i % 5].axis('off')
            ax[i // 5][i % 5].set_title('{}'.format(i), fontsize=6)
        else:
            ax[i // 5][i % 5].axis('off')
    plt.tight_layout()
    
    return fig, ax
