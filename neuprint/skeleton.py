# -*- coding: utf-8 -*-
"""
Functions related to fetching and manipulating skeletons.
"""
from io import StringIO
from itertools import combinations
from collections import namedtuple

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree

from .client import inject_client


@inject_client
def fetch_skeleton(body, heal=False, export_path=None, format='pandas', with_distances=False, *, client=None):
    """
    Equivalent to :py:meth:`.Client.fetch_skeleton()`.  See that function for details.
    """
    return client.fetch_skeleton(body, heal, export_path, format, with_distances)


def skeleton_df_to_nx(df, with_attributes=True, directed=True, with_distances=False, virtual_roots=False, root_dist=np.inf):
    """
    Create a ``networkx.Graph`` from a skeleton DataFrame.

    Args:
        df:
            DataFrame as returned by :py:meth:`.Client.fetch_skeleton()`

        with_attributes:
            If True, store node attributes for x, y, z, radius

        directed:
            If True, return ``nx.DiGraph``, otherwise ``nx.Graph``.
            Edges will point from child to parent.

        with_distances:
            If True, add an edge attribute 'distance' indicating the
            euclidean distance between skeleton nodes.

        virtual_roots:
            If True, include nodes for the 'virtual roots', i.e. node -1.

        root_dist:
            If virtual_roots are requested, this value will be stored as
            the 'distance' of the virtual segment(s) connecting them.

    Returns:
        ``nx.DiGraph`` or ``nx.Graph``
    """
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()

    if with_attributes:
        for row in df.itertuples(index=False):
            g.add_node(row.rowId, x=row.x, y=row.y, z=row.z, radius=row.radius)
    else:
        g.add_nodes_from(df['rowId'].sort_values())

    if not virtual_roots:
        # Instead of assuming that the root node refers to a special parent (e.g. -1),
        # we determine the root_parents by inspection.
        root_parents = pd.Index(df['link'].unique()).difference(df['rowId'].unique())
        root_parents

    if with_distances:
        edges_df = df[['rowId', 'link']].copy()
        edges_df['distance'] = calc_segment_distances(df, root_dist)
        if not virtual_roots:
            edges_df = edges_df.query('link not in @root_parents')
        edges_df = edges_df.sort_values(['rowId', 'link'])
        g.add_weighted_edges_from(edges_df.itertuples(index=False), 'distance')
    else:
        if not virtual_roots:
            edges_df = df.query('link not in @root_parents')
        edges_df = edges_df[['rowId', 'link']]
        edges_df = edges_df.sort_values(['rowId', 'link'])
        g.add_edges_from(edges_df.values)

    return g


def calc_segment_distances(df, root_dist=np.inf):
    """
    For each node (row) in the given skeleton DataFrame,
    compute euclidean distance from the node to its parent (link) node.

    Args:
        df:
            DataFrame as returned by :py:meth:`.Client.fetch_skeleton()`

        root_dist:
            By default, root nodes (i.e. when link == -1) will be assigned a distance of np.inf,
            but you can override that with this setting (e.g. to 0.0).

    Returns:
        np.ndarray
    """
    # Append parent (link) columns to each row by matching
    # each row's 'link' ID with the parent's 'rowId'.
    edges_df = df[['rowId', 'link', *'xyz']].merge(
        df[['rowId', *'xyz']], 'left',
        left_on='link', right_on='rowId', suffixes=['', '_link'])

    diff = edges_df[[*'xyz']] - edges_df[['x_link', 'y_link', 'z_link']].values
    distances = np.linalg.norm(diff, axis=1).astype(np.float32)
    distances[np.isnan(distances)] = root_dist
    return distances


def distances_from_root(df):
    """
    Calculate the distance from the root node(s) to all nodes in the skeleton.
    Return those distances as a new column in the skeleton DataFrame.

    All root nodes will be used, as long as they all have virtual root of -1.

    Args:
        df:
            DataFrame as returned by :py:meth:`.Client.fetch_skeleton()`

    Returns:
        DataFrame
    """
    g = skeleton_df_to_nx(df, directed=False, with_distances=True, virtual_roots=True, root_dist=0.0)
    d = nx.shortest_path_length(g, -1, weight='distance')
    d = pd.Series(d, name='distance').rename_axis('rowId')
    df = df.merge(d, 'left', on='rowId')
    return df


def skeleton_swc_to_df(swc):
    """
    Create a DataFrame from and SWC file.
    The 'node_type' column is discarded.

    Args:
        swc:
            Either a filepath ending in '.swc', or a file object,
            or the contents of an SWC file (as a string).

    Returns:
        ``pd.DataFrame``
    """
    if hasattr(swc, 'read'):
        swc = swc.read()
    else:
        assert isinstance(swc, str)
        if swc.endswith('.swc'):
            with open(swc, 'r') as f:
                swc = f.read()

    cols = ['rowId', 'node_type', 'x', 'y', 'z', 'radius', 'link']
    lines = swc.split('\n')
    lines = filter(lambda line: '#' not in line, lines)
    swc_csv = '\n'.join(lines)

    # Compact dtypes save RAM when loading lots of skeletons
    dtypes = {
        'rowId': np.int32,
        'node_type': np.int8,
        'x': np.float32,
        'y': np.float32,
        'z': np.float32,
        'radius': np.float32,
        'link': np.int32,
    }
    df = pd.read_csv(StringIO(swc_csv), delimiter=' ', engine='c', names=cols, dtype=dtypes, header=None)
    df = df.drop(columns=['node_type'])
    return df


def skeleton_df_to_swc(df, export_path=None):
    """
    Create an SWC file from a skeleton DataFrame.

    Args:
        df:
            DataFrame, as returned by :py:meth:`.Client.fetch_skeleton()`

        export_path:
            Optional. Write the SWC file to disk a the given location.

    Returns:
        ``str``
    """
    df = df.copy()
    df['node_type'] = 0
    df = df[['rowId', 'node_type', 'x', 'y', 'z', 'radius', 'link']]
    swc = "# "
    swc += df.to_csv(sep=' ', header=True, index=False)

    if export_path:
        with open(export_path, 'w') as f:
            f.write(swc)

    return swc


def heal_skeleton(skeleton_df, max_distance=np.inf, root_parent=None):
    """
    Attempt to repair a fragmented skeleton into a single connected component.

    Rather than a single tree, skeletons from neuprint sometimes
    consist of multiple fragments, i.e. multiple connected
    components.  That's due to artifacts in the underlying
    segmentation from which the skeletons were generated.
    In such skeletons, there will be multiple 'root' nodes
    (SWC rows where ``link == -1``).

    This function 'heals' a fragmented skeleton by joining its
    fragments into a single tree.

    First, each fragment is joined to every other fragment at
    their nearest points. The resulting graph has unnecessary
    edges, which are then removed by extracting the minimum
    spanning tree.  The MST is returned as the healed skeleton.

    Args:
        skeleton_df:
            DataFrame as returned by :py:meth:`.Client.fetch_skeleton()`

        max_distance:
            If a skeleton's fragments are very spatially distant, it may
            not be desirable to connect them with a new edge.
            This parameter specifies the maximum length of new edges
            introduced by the healing procedure.  If a skeleton fragment
            cannot be connected to the rest of the skeleton because it's
            too far away, the skeleton will remain fragmented.

        root_parent:
            Typically, root nodes point to a special ID, e.g. -1.
            If this ID is known, then it saves us time when deciding
            if no healing is necessary.

    Returns:
        DataFrame, with ``link`` column updated with updated edges.
    """
    if max_distance is True:
        max_distance = np.inf

    if not max_distance:
        max_distance = 0.0

    if root_parent is None:
        root_parent = -1
    else:
        # Fast path to exit early if we can easily check the number of roots.
        num_roots = (skeleton_df['link'] == root_parent).sum()
        if num_roots == 1:
            # There's only one root and therefore only one component.
            # No healing necessary.
            return skeleton_df

    skeleton_df = skeleton_df.sort_values('rowId', ignore_index=True)
    g = skeleton_df_to_nx(skeleton_df, False, False)

    # Extract each fragment's rows and construct a KD-Tree
    Fragment = namedtuple('Fragment', ['frag_id', 'df', 'kd'])
    fragments = []
    for frag_id, cc in enumerate(nx.connected_components(g)):
        if len(cc) == len(skeleton_df):
            # There's only one component -- no healing necessary
            return skeleton_df
        df = skeleton_df.query('rowId in @cc')
        kd = cKDTree(df[[*'xyz']].values)
        fragments.append( Fragment(frag_id, df, kd) )

    # Sort from big-to-small, so the calculations below use a
    # KD tree for the larger point set in every fragment pair.
    fragments = sorted(fragments, key=lambda frag: -len(frag.df))

    # We could use the full graph and connect all
    # fragments to their nearest neighbors within other fragments,
    # but it's faster to treat each whole fragment as
    # a single node and run MST on that quotient graph,
    # which is tiny.
    frag_graph = nx.Graph()
    for frag_a, frag_b in combinations(fragments, 2):
        coords_b = frag_b.df[[*'xyz']].values
        distances, indexes = frag_a.kd.query(coords_b)

        index_b = np.argmin(distances)
        index_a = indexes[index_b]

        node_a = frag_a.df['rowId'].iloc[index_a]
        node_b = frag_b.df['rowId'].iloc[index_b]
        dist_ab = distances[index_b]

        # Add edge from one fragment to another,
        # but keep track of which fine-grained skeleton
        # nodes were used to calculate distance.
        frag_graph.add_edge( frag_a.frag_id, frag_b.frag_id,
                             node_a=node_a, node_b=node_b,
                             distance=dist_ab )

    # Compute inter-fragment MST edges
    frag_edges = nx.minimum_spanning_edges(frag_graph, weight='distance', data=True)

    # For each inter-fragment edge, add the corresponding
    # fine-grained edge between skeleton nodes in the original graph.
    omit_edges = []
    for _u, _v, d in frag_edges:
        g.add_edge(d['node_a'], d['node_b'])
        if d['distance'] > max_distance:
            omit_edges.append((d['node_a'], d['node_b']))

    # Traverse in depth-first order to compute edges for final tree
    root = skeleton_df['rowId'].iloc[0]

    # Replace 'link' (parent) column using MST edges
    _reorient_skeleton(skeleton_df, root, root_parent, g=g)
    assert (skeleton_df['link'] == root_parent).sum() == 1
    assert skeleton_df['link'].iloc[0] == root_parent

    # Delete edges that violated max_distance
    if omit_edges:
        # Make sure this is invariant to edge direction (check both directions).
        omit_edges = omit_edges + [(b, a) for (a, b) in omit_edges]
        omit_df = pd.DataFrame(omit_edges, columns=['rowId', 'link'])
        omit_df['omit_link'] = -1

        # Remove links for omitted edges (convert child node to a new root).
        skeleton_df = skeleton_df.merge(omit_df, 'left', on=['rowId', 'link'])

        # Update the 'link' column using 'omit_link'
        skeleton_df.update(skeleton_df['omit_link'].rename('link'))
        del skeleton_df['omit_link']

    return skeleton_df


def _reorient_skeleton(skeleton_df, root, root_parent=-1, g=None):
    """
    Replace the 'link' column in each row of the skeleton dataframe
    so that its parent corresponds to a depth-first traversal from
    the given root node.

    Args:
        skeleton_df:
            A skeleton dataframe

        root:
            A rowId to use as the new root node

        g:
            Optional. A nx.Graph representation of the skeleton

    Works in-place.
    """
    g = g or skeleton_df_to_nx(skeleton_df, False, False)
    assert isinstance(g, nx.Graph) and not isinstance(g, nx.DiGraph), \
        "skeleton graph must be undirected"

    edges = list(nx.dfs_edges(g, source=root))

    # If the graph has more than one connected component,
    # the remaining components have arbitrary roots
    if len(edges) != len(g.edges):
        for cc in nx.connected_components(g):
            if root not in cc:
                edges += list(nx.dfs_edges(g, source=cc.pop()))

    edges = pd.DataFrame(edges, columns=['link', 'rowId'])  # parent, child
    edges = edges.set_index('rowId')['link']

    # Replace 'link' (parent) column using DFS edges
    skeleton_df['link'] = skeleton_df['rowId'].map(edges).fillna(root_parent).astype(int)


def reorient_skeleton(skeleton_df, rowId=None, xyz=None, use_max_radius=False):
    """
    Change the root node of a skeleton.

    In general, the root node of the skeletons stored in neuprint is
    not particularly significant, so the directionality of the nodes
    (parent to child or vice-versa) on any given neuron branch is arbitrary.

    This function allows you to pick a different root node and reorient
    the tree with respect to that node. Replaces the 'link' column in
    each row of the skeleton dataframe so that its parent corresponds
    to a depth-first traversal from the new root node.

    You can specify the new root node either by its row, or by a coordinate
    (the closest node to that coordinate will be selected) or by size
    (the largest node will be selected).

    Works in-place.  Only the 'link' column is changed.

    If the given skeleton has more than one connected component (and thus
    more than one root node), the orientation of the edges in other components
    will be arbitrary.

    Args:
        skeleton_df:
            A skeleton dataframe, e.g. as returned by `py:func:fetch_skeleton(..., heal=True)`

        rowId:
            A rowId to use as the new root node

        xyz:
            If given, chooses the node closest to the given coordinate as the new root node.

        use_max_radius:
            If True, choose the largest node (by radius) to use as the new root node.
    """
    assert rowId != 0, \
        "rowId is never 0 in NeuTu skeletons"

    assert bool(rowId) + (xyz is not None) + use_max_radius == 1, \
        "Select either a rowId to use as the new root, or a coordinate, or use_max_radius=True"

    if xyz is not None:
        # Find closest node to the given coordinate
        distances = np.linalg.norm(skeleton_df[[*'xyz']] - xyz, axis=1)
        rowId = skeleton_df['rowId'].iloc[np.argmin(distances)]
    elif use_max_radius:
        # Find the node with the largest radius
        idx = skeleton_df['radius'].idxmax()
        rowId = skeleton_df.loc[idx, 'rowId']

    assert rowId is not None, "You must specify a new root node"

    _reorient_skeleton(skeleton_df, rowId)


def skeleton_segments(skeleton_df):
    """
    Compute a table of skeleton segments.

    A skeleton dataframe is a table of nodes (points) and their parent nodes.

    This function computes a table of segments, where each row lists both the
    child and parent point, along with some attributes describing the segment:
    length, average radius, and segment volume.
    """
    assert isinstance(skeleton_df, pd.DataFrame)

    segment_df = _skeleton_segments(skeleton_df)
    segment_df['avg_radius'] = segment_df.eval('(radius + radius_parent) / 2')

    # Volume of a truncated cone:
    # V = π * h * (R² * r² + R*r) / 3
    PI = np.pi  # noqa
    e = '@PI * length * (radius_parent**2 + radius**2 + radius*radius_parent) / 3'
    segment_df['volume'] = segment_df.eval(e)

    return segment_df


def _skeleton_segments(skeleton_df):
    """
    Compute the table of child-to-parent points and segment lengths.
    """
    segment_df = skeleton_df.merge(skeleton_df[['rowId', 'link', *'xyz', 'radius']],
                                   'inner',
                                   left_on='link',
                                   right_on='rowId',
                                   suffixes=['', '_parent'])

    child_points = segment_df[[*'xyz']].values
    parent_points = segment_df[['x_parent', 'y_parent', 'z_parent']].values
    segment_df['length'] = np.linalg.norm(child_points - parent_points, axis=1)
    return segment_df


def upsample_skeleton(skeleton_df, max_segment_length):
    """
    Insert new nodes into a skeleton make it "higher resolution".
    For all child-parent segments with length greater than the given
    maximum length, subdivide each segment into N smaller equal-length
    segments, such that all of the new segments are (ideally) not
    larger than the given max.

    The 'radius' column is interpolated between the child and parent
    radius values.

    Note:
        By default, skeletons use float32 to store point coordinates.
        Due to rounding errors, the final segment lengths after upsampling
        may still be slightly larger than the requested max!
        If you need better precision, cast your skeleton coordinates
        to float64 first:

        .. code-block:: python

            sk = sk.astype({k: np.float64 for k in 'xyz'})

    Returns:
        DataFrame.  In the result, all previously existing nodes will
        retain their original rowIds and coordinates, but their 'link'
        (parent) and radii may have changed.
    """
    if len(skeleton_df) in (0, 1) or (skeleton_df['link'] == -1).all():
        # Can't upsample a skeleton with no child-parent segments
        return skeleton_df

    seg_df = _skeleton_segments(skeleton_df)
    seg_df = seg_df.loc[seg_df['length'] > max_segment_length]

    if len(seg_df) == 0:
        return skeleton_df

    I0 = seg_df['rowId']
    I1 = seg_df['rowId_parent']
    next_id = 1 + skeleton_df['rowId'].max()

    # It's best to minimize the number of times we call np.linspace(),
    # so we interpolate points and radii in conjunction with a single array.
    PR0 = seg_df[[*'xyz', 'radius']].values
    PR1 = seg_df[['x_parent', 'y_parent', 'z_parent', 'radius_parent']].values

    D = seg_df['length']

    segment_nodes = []
    for i0, i1, pr0, pr1, d in zip(I0, I1, PR0, PR1, D):
        # Number of nodes from child (i0) to parent (i1)
        # excluding the parent (which we won't edit).
        n = int(np.ceil(d / max_segment_length))

        # IDs of the original child and new intermediates going towards
        # the original parent, but not the parent itself.
        I = [i0, *range(next_id, next_id + n - 1)]  # noqa
        next_id += n - 1

        # 'link' (parent id) for the original child and new intermediates
        L = I[1:] + [i1]

        # Interpolate points and radii
        PR = np.linspace(pr0, pr1, n, endpoint=False)

        assert len(PR) == len(I) == len(L)
        segment_nodes.append((I, *PR.T, L))

    segment_cols = [*zip(*segment_nodes)]
    full_cols = [np.concatenate(a) for a in segment_cols]
    new_df = pd.DataFrame(dict(zip(['rowId', *'xyz', 'radius', 'link'], full_cols)))

    # Expand the DataFrame to make room for the new rows,
    # then copy them over.
    all_rowIds = np.sort(pd.concat((skeleton_df['rowId'], new_df['rowId'])).unique())
    dtypes = skeleton_df.dtypes
    skeleton_df = skeleton_df.set_index('rowId').reindex(all_rowIds)
    skeleton_df.update(new_df.set_index('rowId'))

    # Restore to standard column form.
    return skeleton_df.reset_index().astype(dtypes)


def attach_synapses_to_skeleton(skeleton_df, synapses_df):
    """
    Attach a neuron's synapses to its skeleton as new skeleton nodes.
    Synapses are attached to their nearest skeleton node (in euclidean terms).

    Note:
        Skeletons are often "low resolution" so some nodes can be relatively far apart.
        As a consequence, the nearest node to a given synapse may not be close
        to the nearest point on the nearest line segment to the synapse.
        To combat this problem, see `py:func:upsample_skeleton()`.

    Args:
        skeleton_df:
            A DataFrame containing a neuron skeleton,
            as produced by ``fetch_skeleton()``

        synapses_df:
            A DataFrame of synapse points, as produced by ``fetch_synapses()``.

    Returns:
        DataFrame
        Rows are appended to the skeleton (one per synapse), and a new column is
        added to distinguish between nodes which belonged to the original skeleton
        ('neurite') and those which were added for synapses ('pre' or 'post').

    Example:

        .. code-block:: ipython

                In [4]: from neuprint import fetch_skeleton, fetch_synapses, attach_synapses_to_skeleton
                   ...: body = 1136399017
                   ...: skeleton = fetch_skeleton(body, heal=True)
                   ...: synapses = fetch_synapses(body)
                   ...: attach_synapses_to_skeleton(skeleton, synapses)

                Out[4]:
                    rowId             x             y             z     radius  link structure
                0         1  12798.000000  30268.000000  15812.000000  21.000000    -1   neurite
                1         2  12746.700195  30370.699219  15788.700195  55.464100     1   neurite
                2         3  12727.200195  30411.199219  15767.599609  68.081200     2   neurite
                3         4  12705.299805  30475.599609  15716.400391  58.952702     3   neurite
                4         5  12687.400391  30499.500000  15692.500000  50.619999     4   neurite
                ...     ...           ...           ...           ...        ...   ...       ...
                2032   2033  12073.000000  32575.000000  14386.000000   0.000000   651      post
                2033   2034  10072.000000  32572.000000  14464.000000   0.000000   685      post
                2034   2035  10647.000000  31797.000000  14057.000000   0.000000   760      post
                2035   2036  11203.000000  30673.000000  14839.000000   0.000000  1086      post
                2036   2037  11116.000000  30613.000000  14707.000000   0.000000  1068      post

                [2037 rows x 7 columns]
    """
    skeleton_df = skeleton_df.copy(deep=False).reset_index(drop=True)
    synapses_df = synapses_df.copy(deep=False).reset_index(drop=True)

    skeleton_df['structure'] = 'neurite'
    synapses_df['structure'] = synapses_df['type']
    synapses_df['radius'] = 0.0

    kd = cKDTree(skeleton_df[[*'xyz']].values)
    _, indexes = kd.query(synapses_df[[*'xyz']].values)

    synapses_df['link'] = skeleton_df.loc[indexes, 'rowId'].values
    synapses_df['rowId'] = synapses_df.index + skeleton_df['rowId'].max() + 1

    relevant_cols = ['rowId', *'xyz', 'radius', 'link', 'structure']
    synapses_df = synapses_df[relevant_cols]
    skeleton_df = skeleton_df[relevant_cols]

    combined = pd.concat((skeleton_df, synapses_df), ignore_index=True)
    combined['structure'] = pd.Categorical(combined['structure'])
    return combined
