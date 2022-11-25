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
            method: bool = 'matplotlib',
            cmap = None,
            fontsize: int = 6,
            max_columns: int = 4,
            marker_size: float = 3.,
            markerscale: float = 1.,
            vmin = None,
            vmax = None,
            axes_visible: bool = False,
            legend: bool = True):
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
        
    if x_key not in list(adata.obsm.keys()):
        raise(NameError('No embedding found to visualize in adata.obsm["{}"].'.format(x_key)))
        
    if max_columns < 2:
        raise(ValueError('max_columns must be > 1.'))
        
    colors_obs = True if color_key in list(adata.obs.keys()) else False
    colors_obsm = True if color_key in list(adata.obsm.keys()) else False
    
    if colors_obs and colors_obsm:
        raise(NameError('Found adata.obs["{}"] and adata.obsm["{}"]. Please make keys unique and select the one you meant.'.format(color_key, color_key)))
        
    n_dim = adata.obsm[x_key].shape[1]
        
    if method == 'matplotlib':
        if n_dim == 2:
            cmap = 'Spectral' if cmap is None else cmap
            cmap = mpl.cm.get_cmap(cmap) if type(cmap) == str else cmap
            
            n_plots = 1 if (colors_obs or not (colors_obs or colors_obsm)) else adata.obsm[color_key].shape[1]
            n_rows = n_plots // max_columns if n_plots % max_columns == 0 else n_plots // max_columns + 1
            n_columns = max_columns if n_plots > max_columns else n_plots
            
            df = pd.DataFrame(adata.obsm[x_key], columns=['x', 'y'])
            
            if colors_obs:
                color_names = [color_key]
                df[color_key] = list(adata.obs[color_key])
                
            elif colors_obsm:
                if isinstance(adata.obsm[color_key], pd.DataFrame):
                    color_names = adata.obsm[color_key].columns
                    for name in color_names:
                        df[name] = list(adata.obsm[color_key][name])
                else:
                    color_names = [i for i in range(n_plots)]
                    for i in range(n_plots):
                        df[i] = list(adata.obsm[color_key][:, i])
            else:
                color_names = ['color']
                df['color'] = ['Undefined' for _ in range(adata.shape[0])]
            
            df = df.sample(frac=1)
            
            fig, ax = plt.subplots(n_rows, n_columns)
            
            for i in range(n_rows * n_columns):
                row, col = i // n_columns, i % n_columns
                
                if n_plots == 1:
                    if not axes_visible:
                        ax.axis('off')
                    ax.set_aspect('equal')
                elif n_rows == 1:
                    if not axes_visible:
                        ax[col].axis('off')
                    ax[col].set_aspect('equal')
                else:
                    if not axes_visible:
                        ax[row][col].axis('off')
                    ax[row][col].set_aspect('equal')
                
                if i < n_plots:
                    is_discrete = False
                    for el in df[color_names[i]]:
                        if type(el) is str: 
                            is_discrete = True
                            break
                    
                    if not is_discrete: 
                        if n_plots == 1:
                            scatter = ax.scatter(x=df['x'],
                                                 y=df['y'], 
                                                 s=marker_size, 
                                                 c=df[color_names[i]], 
                                                 cmap=cmap, 
                                                 edgecolor='none',
                                                 vmin=np.percentile(df[color_names[i]], 1) if vmin is None else vmin, 
                                                 vmax=np.percentile(df[color_names[i]], 99) if vmax is None else vmax)
                        elif n_rows == 1:
                            scatter = ax[col].scatter(x=df['x'],
                                                      y=df['y'], 
                                                      s=marker_size, 
                                                      c=df[color_names[i]], 
                                                      cmap=cmap, 
                                                      edgecolor='none',
                                                      vmin=np.percentile(df[color_names[i]], 1) if vmin is None else vmin, 
                                                      vmax=np.percentile(df[color_names[i]], 99) if vmax is None else vmax)
                        else:
                            scatter = ax[row][col].scatter(x=df['x'],
                                                           y=df['y'], 
                                                           s=marker_size, 
                                                           c=df[color_names[i]], 
                                                           cmap=cmap, 
                                                           edgecolor='none',
                                                           vmin=np.percentile(df[color_names[i]], 1) if vmin is None else vmin, 
                                                           vmax=np.percentile(df[color_names[i]], 99) if vmax is None else vmax)
                            
                        scalarmappaple = mpl.cm.ScalarMappable(
                            norm=mpl.colors.Normalize(vmin=np.percentile(df[color_names[i]], 1) if vmin is None else vmin, 
                                                      vmax=np.percentile(df[color_names[i]], 99) if vmax is None else vmax), 
                            cmap=cmap)
                        
                        if legend:
                            if n_plots == 1:
                                ax.set_title(color_names[i], fontsize=fontsize)
                                cbar = fig.colorbar(scatter, ax=ax, fraction=0.04)
                            elif n_rows == 1:
                                ax[col].set_title(color_names[i], fontsize=fontsize)
                                cbar = fig.colorbar(scatter, ax=ax[col], fraction=0.04)
                            else:
                                ax[row][col].set_title(color_names[i], fontsize=fontsize)
                                cbar = fig.colorbar(scatter, ax=ax[row][col], fraction=0.04)

                            cbar.ax.tick_params(labelsize=fontsize, length=0)
                            cbar.outline.set_color('white')
                    
                    else:
                        types = np.unique(df[color_names[i]])
                        d = {t: i for i, t in enumerate(types)}
                        df['c'] = [cmap(d[el]/(len(d.keys())-1)) for el in df[color_names[i]]]
                        
                        patches = [Line2D(range(1), range(1), color="white", marker='o', 
                                          markerfacecolor=cmap(d[t]/(len(d.keys())-1)), label=t) for t in d]
                        
                        if n_plots == 1:
                            ax.scatter(x=df['x'],
                                       y=df['y'],
                                       s=marker_size, 
                                       c=df['c'], 
                                       edgecolor='none')
                            if legend:
                                ax.legend(handles=patches, 
                                          fontsize=fontsize, 
                                          borderpad=0, 
                                          frameon=False, 
                                          markerscale=markerscale)

                        elif n_rows == 1:
                            ax[col].scatter(x=df['x'],
                                            y=df['y'],
                                            s=marker_size, 
                                            c=df['c'], 
                                            edgecolor='none')
                            if legend:
                                ax[col].legend(handles=patches, 
                                               fontsize=fontsize, 
                                               borderpad=0, 
                                               frameon=False, 
                                               markerscale=markerscale)
                        else:
                            ax[row][col].scatter(x=df['x'],
                                                 y=df['y'],
                                                 s=marker_size, 
                                                 c=df['c'], 
                                                 edgecolor='none')
                            if legend:
                                ax[row][col].legend(handles=patches, 
                                                    fontsize=fontsize, 
                                                    borderpad=0, 
                                                    frameon=False, 
                                                    markerscale=markerscale)
                        
            return fig, ax
            
        else:
            raise(ValueError('When method="matplotlib", visualized embedding must be 2D. adata.obsm["{}"] is {}D.'.format(x_key, n_dim)))
                
    elif method == 'plotly':
        if colors_obsm:
            raise(NameError('When method="plotly", color_key best be a adata.obs key. You passed adata.obsm key.'))
        
        if n_dim == 2:
            
            df = pd.DataFrame(adata.obsm[x_key], columns=['x', 'y'])
            df['color'] = list(adata.obs[color_key]) if colors_obs else ['Undefined' for _ in range(adata.shape[0])]
            df = df.sample(frac=1)

            fig = px.scatter(df, 
                             x='x',
                             y='y', 
                             color='color', 
                             hover_name='color', 
                             hover_data={'x': False, 'y': False, 'color': False},
                             range_color=[vmin, vmax] if ((vmin is not None) and (vmax is not None)) else None,
                             color_continuous_scale=cmap,
                             title=color_key)

            fig.update_layout(scene = dict(
                xaxis = dict(
                    backgroundcolor='white',
                    visible=axes_visible, 
                    showticklabels=axes_visible,
                    gridcolor='#bcbcbc',
                    showbackground=True,
                    zerolinecolor='white'),
                yaxis = dict(
                    backgroundcolor='white',
                    visible=axes_visible, 
                    showticklabels=axes_visible,
                    gridcolor='#bcbcbc',
                    showbackground=True,
                    zerolinecolor='white')))
            
            fig.update_layout({
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white'})
            
        elif n_dim == 3:
            
            df = pd.DataFrame(adata.obsm[x_key], columns=['x', 'y', 'z'])
            df['color'] = list(adata.obs[color_key]) if colors_obs else ['Undefined' for _ in range(adata.shape[0])]
            df = df.sample(frac=1)

            fig = px.scatter_3d(df, 
                                x='x', 
                                y='y',
                                z='z', 
                                color='color', 
                                hover_name='color', 
                                hover_data={'x': False, 'y': False, 'z': False, 'color': False},
                                range_color=[vmin, vmax] if ((vmin is not None) and (vmax is not None)) else None,
                                color_continuous_scale=cmap, 
                                title=color_key)

            fig.update_layout(scene = dict(
                xaxis = dict(
                    backgroundcolor='white',
                    visible=axes_visible, 
                    showticklabels=axes_visible,
                    gridcolor='#bcbcbc',
                    showbackground=True,
                    zerolinecolor='white'),
                yaxis = dict(
                    backgroundcolor='white',
                    visible=axes_visible, 
                    showticklabels=axes_visible,
                    gridcolor='#bcbcbc',
                    showbackground=True,
                    zerolinecolor='white'),
                zaxis = dict(
                    backgroundcolor='white',
                    visible=axes_visible, 
                    showticklabels=axes_visible,
                    gridcolor='#bcbcbc',
                    showbackground=True,
                    zerolinecolor='white')))
            
            fig.update_layout({
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white'})
            
        else:
            raise(ValueError('When method="plotly", visualized embedding must be 2D or 3D. adata.obsm["{}"] is {}D.'.format(x_key, n_dim)))

        fig.update_traces(marker=dict(size=marker_size), selector=dict(mode='markers'))
        fig.update_layout(legend={'itemsizing': 'constant'})

        return fig
    else:
        raise(NameError('Valid plotting methods: matplotlib, plotly. You passed {}.'.format(method)))
