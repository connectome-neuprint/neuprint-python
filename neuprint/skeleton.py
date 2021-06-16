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


def skeleton_df_to_nx(df, with_attributes=True, directed=True, with_distances=False):
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

    if with_distances:
        edges_df = df[['rowId', 'link']].copy()
        edges_df['distance'] = calc_segment_distances(df)
        edges_df = edges_df.query('link != -1').sort_values(['rowId', 'link'])
        g.add_weighted_edges_from(edges_df.itertuples(index=False), 'distance')
    else:
        edges_df = df.query('link != -1')[['rowId', 'link']]
        edges_df = edges_df.sort_values(['rowId', 'link'])
        g.add_edges_from(edges_df.values)

    return g


def calc_segment_distances(df):
    """
    For each node (row) in the given skeleton DataFrame,
    compute euclidean distance from the node to its parent (link) node.
    Root nodes (i.e. when link == -1) will be assigned a distance of np.inf.

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
    distances[np.isnan(distances)] = np.inf
    return distances


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


def heal_skeleton(skeleton_df, max_distance=np.inf):
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

    Returns:
        DataFrame, with ``link`` column updated with updated edges.
    """
    if max_distance is True:
        max_distance = np.inf

    if not max_distance:
        max_distance = 0.0

    skeleton_df = skeleton_df.sort_values('rowId').reset_index(drop=True)
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
    # fragment pairs at their nearest neighbors,
    # but it's faster to treat each fragment as a
    # single node and run MST on that quotient graph,
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
    _reorient_skeleton(skeleton_df, root, g)
    assert (skeleton_df['link'] == -1).sum() == 1
    assert skeleton_df['link'].iloc[0] == -1

    # Delete edges that violated max_distance
    for a,b in omit_edges:
        q = '(rowId == @a and link == @b) or (rowId == @b and link == @a)'
        idx = skeleton_df.query(q).index
        skeleton_df.loc[idx, 'link'] = -1

    return skeleton_df


def _reorient_skeleton(skeleton_df, root, g=None):
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
    skeleton_df['link'] = skeleton_df['rowId'].map(edges).fillna(-1).astype(int)


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
