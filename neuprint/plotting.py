"""
Miscellaneous plotting functions.


Note:
    These functions require additional dependencies,
    which aren't listed by default dependencies of neuprint-python.
    (See docs for each function.)
"""
import numpy as np
import pandas as pd

from .client import inject_client
from .skeleton import skeleton_df_to_nx

def plot_soma_projections(neurons_df, color_by='cellBodyFiber'):
    """
    Plot the soma locations as XY, XZ, and ZY 2D projections,
    colored by the given column.

    Requires ``bokeh``.

    Returns a layout which can be displayed
    with ``bokeh.plotting.show()``.

    Example:

    .. code-block: python

        from neuprint import fetch_neurons, NeuronCriteria as NC
        from bokeh.plotting import output_notebook
        output_notebook()

        criteria = NC(status='Traced', cropped=False)
        neurons_df, _roi_counts_df = fetch_neurons(criteria)
        p = plot_soma_projections(neurons_df, 'cellBodyFiber')
        show(p)

    """
    import bokeh
    from bokeh.plotting import figure
    from bokeh.layouts import gridplot

    neurons_df = neurons_df[['somaLocation', color_by]].copy()

    extract_soma_coords(neurons_df)
    assign_colors(neurons_df, color_by)

    neurons_with_soma_df = neurons_df.query('not somaLocation.isnull()')
    def soma_projection(axis1, axis2, flip1, flip2):
        x = neurons_with_soma_df[f'soma_{axis1}'].values
        y = neurons_with_soma_df[f'soma_{axis2}'].values
        p = figure(title=f'{axis1}{axis2}')
        p.scatter(x, y, color=neurons_with_soma_df['color'])
        p.x_range.flipped = flip1
        p.y_range.flipped = flip2
        p.toolbar.logo = None
        return p

    p_xy = soma_projection('x', 'y', False, True)
    p_xz = soma_projection('x', 'z', False, True)
    p_zy = soma_projection('z', 'y', True, True)

    # This will produce one big plot with a shared toolbar
    layout = gridplot([[p_xy, p_xz], [None, p_zy]])

    # Discard the help buttons and bokeh logo
    tbar = layout.children[0].toolbar
    tbar.logo = None
    tbar.tools = [t for t in tbar.tools if not isinstance(t, bokeh.models.tools.HelpTool)]

    return layout


def plot_soma_3d(neurons_df, color_by='cellBodyFiber', point_size=1.0):
    """
    Plot the soma locations in 3D, colored randomly according
    to the column given in ``color_by``.

    Requires ``ipyvolume``.
    If using Jupyterlab, install it like this:

    .. code-block: bash

        conda install -c conda-forge ipyvolume
        jupyter labextension install ipyvolume

    Example:

        .. code-block: python

            from neuprint import fetch_neurons, NeuronCriteria as NC

            criteria = NC(status='Traced', cropped=False)
            neurons_df, _roi_counts_df = fetch_neurons(criteria)
            plot_soma_3d(neurons_df, 'cellBodyFiber')
    """
    import ipyvolume.pylab as ipv
    neurons_df = neurons_df[['somaLocation', color_by]].copy()

    extract_soma_coords(neurons_df)
    assign_colors(neurons_df, color_by)

    neurons_with_soma_df = neurons_df.query('not somaLocation.isnull()')
    assert neurons_with_soma_df.eval('color.isnull()').sum() == 0

    soma_x = neurons_with_soma_df['soma_x'].values
    soma_y = neurons_with_soma_df['soma_y'].values
    soma_z = neurons_with_soma_df['soma_z'].values

    def color_to_vals(color_string):
        # Convert bokeh color string into float tuples,
        # e.g. '#00ff00' -> (0.0, 1.0, 0.0)
        s = color_string
        return (int(s[1:3], 16) / 255,
                int(s[3:5], 16) / 255,
                int(s[5:7], 16) / 255 )

    color_vals = neurons_with_soma_df['color'].apply(color_to_vals).tolist()

    # DVID coordinate system assumes (0,0,0) is in the upper-left.
    # For consistency with DVID and neuroglancer conventions,
    # we invert the Y and X coordinates.
    ipv.figure()
    ipv.scatter(soma_x, -soma_y, -soma_z, color=color_vals, marker="circle_2d", size=point_size)
    ipv.show()


@inject_client
def plot_skeleton_3d(skeleton, color='blue', *, client=None):
    """
    Plot the given skeleton in 3D.

    Args:
        skeleton:
            Either a bodyId or a pre-fetched pandas DataFrame

        color:
            See ``ipyvolume`` docs.
            Examples: ``'blue'``, ``'#0000ff'``
            If the skeleton is fragmented, you can give a list
            of colors and each fragment will be shown in a
            different color.

    Requires ``ipyvolume``.
    If using Jupyterlab, install it like this:

    .. code-block: bash

        conda install -c conda-forge ipyvolume
        jupyter labextension install ipyvolume
    """
    import ipyvolume.pylab as ipv

    if np.issubdtype(type(skeleton), np.integer):
        skeleton = client.fetch_skeleton(skeleton, format='pandas')

    assert isinstance(skeleton, pd.DataFrame)
    g = skeleton_df_to_nx(skeleton)

    def skel_path(root):
        """
        We want to plot each skeleton fragment as a single continuous line,
        but that means we have to backtrack: parent -> leaf -> parent
        to avoid jumping from one branch to another.
        This means that the line will be drawn on top of itself,
        and we'll have 2x as many line segments in the plot,
        but that's not a big deal.
        """
        def accumulate_points(n):
            p = (g.nodes[n]['x'], g.nodes[n]['y'], g.nodes[n]['z'])
            points.append(p)

            children = [*g.successors(n)]
            if not children:
                return
            for c in children:
                accumulate_points(c)
                points.append(p)

        points = []
        accumulate_points(root)
        return np.asarray(points)

    # Skeleton may contain multiple fragments,
    # so compute the path for each one.
    def skel_paths(df):
        paths = []
        for root in df.query('link == -1')['rowId']:
            paths.append(skel_path(root))
        return paths

    paths = skel_paths(skeleton)
    if isinstance(color, str):
        colors = len(paths)*[color]
    else:
        colors = (1+len(paths)//len(color))*color

    ipv.figure()
    for points, color in zip(paths, colors):
        ipv.plot(*points.transpose(), color)
    ipv.show()


def extract_soma_coords(neurons_df):
    """
    Expand the ``somaLocation`` column into three separate
    columns for ``soma_x``, ``soma_y``, and ``soma_z``.

    If ``somaLocation is None``, then the soma coords will be ``NaN``.

    Works in-place.
    """
    neurons_df['soma_x'] = neurons_df['soma_y'] = neurons_df['soma_z'] = np.nan

    somaloc = neurons_df.query('not somaLocation.isnull()')['somaLocation']
    somaloc_array = np.asarray(somaloc.tolist())

    neurons_df.loc[somaloc.index, 'soma_x'] = somaloc_array[:, 0]
    neurons_df.loc[somaloc.index, 'soma_y'] = somaloc_array[:, 1]
    neurons_df.loc[somaloc.index, 'soma_z'] = somaloc_array[:, 2]


def assign_colors(neurons_df, color_by='cellBodyFiber'):
    """
    Use a random colortable to assign a color to each row,
    according to the column given in ``color_by``.

    NaN values are always black.

    Works in-place.
    """
    from bokeh.palettes import Turbo256
    colors = list(Turbo256)
    colors[0] = '#000000'
    color_categories = np.sort(neurons_df[color_by].fillna('').unique())
    assert color_categories[0] == ''

    np.random.seed(0)
    np.random.shuffle(color_categories[1:])
    assert color_categories[0] == ''

    while len(colors) < len(color_categories):
        colors.extend(colors[1:])

    color_mapping = dict(zip(color_categories, colors))
    neurons_df['color'] = neurons_df[color_by].fillna('').map(color_mapping)

