import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec


def scatter_interactive(adata: anndata.AnnData,
                        x: str,
                        s: float = 1.,
                        c: str = None,
                        cdiscrete = None,
                        ccontinuous = None,
                        vmin = None,
                        vmax = None,
                        title=None,
                        showlegend: bool = True,
                        showaxes: bool = False,
                        save = None):
    
    """
    2D and 3D interactive scatter plots

    Generates a 2D or 3D interactive scatter plot using Plotly. The plot is based on data 
    stored in `adata.obsm` and can optionally use a color scheme from `adata.obs`.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param x: Key in `adata.obsm` containing 2D or 3D data for plotting.
    :type x: str

    :param s: Size of the scatter plot markers. (default: 1.0)
    :type s: float

    :param c: Key in `adata.obs` specifying the color scheme. (default: `None`)
    :type c: str or None

    :param cdiscrete: A dictionary mapping discrete groups in the color scheme to specific colors. Used for discrete color schemes. (default: `None`)
    :type cdiscrete: dict or None

    :param ccontinuous: Colormap for continuous color schemes. Options from Plotly's `colors.sequential`, `colors.diverging`, or `colors.cyclical`. (default: `None`)
    :type ccontinuous: str or None

    :param vmin: Lower bound of the colormap for continuous color schemes. Must be used with `vmax`. (default: `None`)
    :type vmin: float or None

    :param vmax: Upper bound of the colormap for continuous color schemes. Must be used with `vmin`. (default: `None`)
    :type vmax: float or None

    :param title: Title of the plot. (default: `None`)
    :type title: str or None

    :param showlegend: Whether to display a legend. (default: `True`)
    :type showlegend: bool

    :param showaxes: Whether to display the plot axes. (default: `False`)
    :type showaxes: bool

    :param save: Path to save the plot as an HTML file. If `None`, the plot is returned as a Plotly figure. (default: `None`)
    :type save: str or None

    :returns: 
        - If `save=None`: Returns a Plotly figure object.
        - Otherwise, saves the plot to the specified path as an HTML file.
    :rtype: plotly.graph_objs._figure.Figure or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))
            adata.obsm["embedding"] = np.random.rand(100, 2)
            adata.obs["group"] = np.random.choice(["A", "B", "C"], 100)

            # Generate an interactive scatter plot
            oci.pl.scatter_interactive(adata, x="embedding", c="group", s=5, title="Interactive Scatter Plot")
    """
    
    if x not in list(adata.obsm.keys()):
        raise(NameError('No data found in adata.obsm["{}"].'.format(x)))
        
    ndim = adata.obsm[x].shape[1]
    
    if ndim not in [2, 3]:
        raise(ValueError('Specified array must be 2D or 3D.'))
        
    df = pd.DataFrame(adata.obsm[x], columns=['x', 'y'] if ndim == 2 else ['x', 'y', 'z'])
    
    if c is not None:
        if c not in adata.obs.keys():
            raise(NameError('No data found in adata.obs["{}"].'.format(c)))

    cname = c if c is not None else 'color'
    df[cname] = list(adata.obs[c]) if c is not None else ['Undefined' for _ in range(adata.shape[0])]
    df = df.sample(frac=1)
        
    if ndim == 2:
        fig = px.scatter(df, 
                         x='x',
                         y='y', 
                         color=cname, 
                         hover_name=cname, 
                         hover_data={'x': False, 'y': False, cname: False},
                         range_color=[vmin, vmax] if ((vmin is not None) and (vmax is not None)) else None,
                         color_continuous_scale=ccontinuous,
                         color_discrete_map=cdiscrete,
                         title=c if title is None else title)

        fig.update_layout(scene = dict(
            xaxis = dict(
                backgroundcolor='white',
                visible=showaxes, 
                showticklabels=showaxes,
                gridcolor='#bcbcbc',
                showbackground=True,
                zerolinecolor='white'),
            yaxis = dict(
                backgroundcolor='white',
                visible=showaxes, 
                showticklabels=showaxes,
                gridcolor='#bcbcbc',
                showbackground=True,
                zerolinecolor='white')))
        fig.update_xaxes(visible=showaxes)
        fig.update_yaxes(visible=showaxes)
        
    else:
        fig = px.scatter_3d(df, 
                            x='x', 
                            y='y',
                            z='z', 
                            color=cname, 
                            hover_name=cname, 
                            hover_data={'x': False, 'y': False, 'z': False, cname: False},
                            range_color=[vmin, vmax] if ((vmin is not None) and (vmax is not None)) else None,
                            color_continuous_scale=ccontinuous,
                            color_discrete_map=cdiscrete,
                            title=c if title is None else title)

        fig.update_layout(scene = dict(
            xaxis = dict(
                backgroundcolor='white',
                visible=showaxes, 
                showticklabels=showaxes,
                gridcolor='#bcbcbc',
                showbackground=True,
                zerolinecolor='white'),
            yaxis = dict(
                backgroundcolor='white',
                visible=showaxes, 
                showticklabels=showaxes,
                gridcolor='#bcbcbc',
                showbackground=True,
                zerolinecolor='white'), 
            zaxis = dict(
                backgroundcolor='white',
                visible=showaxes, 
                showticklabels=showaxes,
                gridcolor='#bcbcbc',
                showbackground=True,
                zerolinecolor='white')))
        fig.update_layout(scene=dict(
            xaxis=dict(showticklabels=showaxes),
            yaxis=dict(showticklabels=showaxes),
            zaxis=dict(showticklabels=showaxes)))
        
    fig.update_layout(showlegend=showlegend)
    fig.update_traces(marker=dict(size=s))
    fig.update_layout(legend={'itemsizing': 'constant'})
    fig.update_layout({'plot_bgcolor': 'white', 'paper_bgcolor': 'white'})
    
    if save is not None:
        fig.write_html(save)
    else:
        return fig
