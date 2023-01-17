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
                        markersize: float = 1.,
                        c: str = None,
                        cdiscrete = None,
                        ccontinuous = None,
                        vmin = None,
                        vmax = None,
                        title=None,
                        showlegend: bool = True,
                        showaxes: bool = False,
                        save = None):
    
    """2D and 3D interactive scatter plots
    
    Generates a 2D or 3D scatter plot using Plotly.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    x
        `adata.obsm` key with 2D or 3D data.
    markersize
         The marker size. (default: 1.)
    c
        `adata.obs` key with a color scheme. (default: :obj:`None`)
    cdiscrete
        Applied when color scheme is discrete. Can be a dictionary 
        mapping color scheme groups to colors. (default: :obj:`None`)
    ccontinuous
        Various useful color scales are available in the plotly.express.colors submodules, 
        specifically plotly.express.colors.sequential, plotly.express.colors.diverging and 
        plotly.express.colors.cyclical. (default: :obj:`None`)
    vmin
        Applied when color scheme is continuous. Lower bound of color scheme.
        `vmax` also must be specified. (default: :obj:`None`)
    vmax
        Applied when color scheme is continuous. Upper bound of color scheme. 
        `vmin` also must be specified. (default: :obj:`None`)
    title
        Plot title. (default: :obj:`None`)
    showlegend
        If `True`, legend is displayed. (default: `True`)
    showaxes
        If `True`, axes are displayed. (default: `True`)
    save
        Path for saving the figure as a html file. (default: :obj:`None`)
      
    Returns
    -------
    :class:`plotly.graph_objs._figure.Figure` if `save = None`.
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
    fig.update_traces(marker=dict(size=markersize))
    fig.update_layout(legend={'itemsizing': 'constant'})
    fig.update_layout({'plot_bgcolor': 'white', 'paper_bgcolor': 'white'})
    
    if save is not None:
        fig.write_html(save)
    else:
        return fig
