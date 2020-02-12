import os
import sys
import copy
import collections
from textwrap import indent, dedent

import numpy as np
import pandas as pd
from asciitree import LeftAligned

from .client import inject_client
from .segmentcriteria import SegmentCriteria
from .synapsecriteria import SynapseCriteria
from .utils import make_args_iterable, trange

# ujson is faster than Python's builtin json module
import ujson

NEURON_COLS = ['bodyId', 'instance', 'type',
               'pre', 'post', 'size',
               'status', 'cropped', 'statusLabel',
               'cellBodyFiber',
               'somaRadius', 'somaLocation',
               'inputRois', 'outputRois', 'roiInfo']

@inject_client
def fetch_custom(cypher, dataset="", format='pandas', *, client=None):
    """
    Make a custom cypher query.
    
    Alternative form of :py:meth:`.Client.fetch_custom()`, as a free function.
    That is, ``fetch_custom(..., client=c)`` is equivalent to ``c.fetch_custom(...)``.

    If ``client=None``, the default ``Client`` is used
    (assuming you have created at least one ``Client``.)
    
    Args:
        cypher:
            A cypher query string

        dataset:
            *Deprecated. Please provide your dataset as a Client constructor argument.*
            
            Which neuprint dataset to query against.
            If None provided, the client's default dataset is used.

        format:
            Either 'pandas' or 'json'.
            Whether to load the results into a pandas DataFrame,
            or return the server's raw json response as a Python dict.

        client:
            If not provided, the global default :py:class:`.Client` will be used.
    
    Returns:
        Either json or DataFrame, depending on ``format``.
    """
    return client.fetch_custom(cypher, dataset, format)


@inject_client
def fetch_meta(*, client=None):
    """
    Fetch the dataset metadata.
    Parses json fields as needed.
    
    Returns:
        dict
    
    Example
    
    .. code-block:: ipython
    
        In [1]: meta = fetch_meta()
    
        In [2]: list(meta.keys())
        Out[2]:
        ['dataset',
         'info',
         'lastDatabaseEdit',
         'latestMutationId',
         'logo',
         'meshHost',
         'neuroglancerInfo',
         'neuroglancerMeta',
         'postHPThreshold',
         'postHighAccuracyThreshold',
         'preHPThreshold',
         'primaryRois',
         'roiHierarchy',
         'roiInfo',
         'statusDefinitions',
         'superLevelRois',
         'tag',
         'totalPostCount',
         'totalPreCount',
         'uuid']
    """
    q = """\
        MATCH (m:Meta)
        WITH m as m,
             apoc.convert.fromJsonMap(m.roiInfo) as roiInfo,
             apoc.convert.fromJsonMap(m.roiHierarchy) as roiHierarchy,
             apoc.convert.fromJsonMap(m.neuroglancerInfo) as neuroglancerInfo,
             apoc.convert.fromJsonList(m.neuroglancerMeta) as neuroglancerMeta,
             apoc.convert.fromJsonMap(m.statusDefinitions) as statusDefinitions
        RETURN m as meta, roiInfo, roiHierarchy, neuroglancerInfo, neuroglancerMeta, statusDefinitions
    """
    df = client.fetch_custom(q)
    meta = df['meta'].iloc[0]
    meta.update(df.drop(columns='meta').iloc[0].to_dict())
    return meta


@inject_client
def fetch_all_rois(*, client=None):
    """
    List all ROIs in the dataset.
    """
    meta = fetch_meta(client=client)
    return _all_rois_from_meta(meta)


def _all_rois_from_meta(meta):
    official_rois = {*meta['roiInfo'].keys()}

    # These two ROIs are special:
    # For historical reasons, they exist as tags,
    # but are not listed in the Meta roiInfo.
    hidden_rois = {'FB-column3', 'AL-DC3'}

    return sorted(official_rois | hidden_rois)


@inject_client
def fetch_primary_rois(*, client=None):
    """
    List 'primary' ROIs in the dataset.
    Primary ROIs do not overlap with each other.
    """
    q = "MATCH (m:Meta) RETURN m.primaryRois as rois"
    rois = client.fetch_custom(q)['rois'].iloc[0]
    return sorted(rois)


def fetch_roi_hierarchy(include_subprimary=True, mark_primary=True, format='dict', *, client=None):
    """
    Fetch the ROI hierarchy nesting relationships.
    
    Most ROIs in neuprint are part of a hierarchy of nested regions.
    The structure of the hierarchy is stored in the dataset metadata,
    and can be retrieved with this function.
    
    Args:
        include_subprimary:
            If True, all hierarchy levels are included in the output.
            Otherwise, the hierarchy will only go as deep as necessary to
            cover all "primary" ROIs, but not any sub-primary ROIs that
            are contained within them.
        
        mark_primary:
            If True, append an asterisk (``*``) to the names of
            "primary" ROIs in the hierarchy.
            Primary ROIs do not overlap with each other.
            
        format:
            Either ``"dict"``, ``"text"``, or ``nx``.
            Specifies whether to return the hierarchy as a `dict`, or as
            a printable text-based tree, or as a ``networkx.DiGraph``
            (requires ``networkx``).
    
    Returns:
        Either ``dict``, ``str``, or ``nx.DiGraph``,
        depending on your chosen ``format``.

    Example:
    
        .. code-block:: ipython
        
            In [1]: from neuprint.queries import fetch_roi_hierarchy
               ...:
               ...: # Print the first few nodes of the tree -- you get the idea
               ...: roi_tree_text = fetch_roi_hierarchy(False, True, 'text')
               ...: print(roi_tree_text[:180])
            hemibrain
             +-- AL(L)*
             +-- AL(R)*
             +-- AOT(R)
             +-- CX
             |   +-- AB(L)*
             |   +-- AB(R)*
             |   +-- EB*
             |   +-- FB*
             |   +-- NO*
             |   +-- PB*
             +-- GC
             +-- GF(R)
             +-- GNG*
             +-- INP
             |
    """
    assert format in ('dict', 'text', 'nx')
    meta = fetch_meta(client=client)
    hierarchy = meta['roiHierarchy']
    primary_rois = {*meta['primaryRois']}

    def insert(h, d):
        name = h['name']
        is_primary = (name in primary_rois)
        if mark_primary and is_primary:
            name += "*"
        
        d[name] = {}
        
        if 'children' not in h:
            return

        if is_primary and not include_subprimary:
            return
        
        for c in sorted(h['children'], key=lambda c: c['name']):
            insert(c, d[name])

    d = {}
    insert(hierarchy, d)
    
    if format == 'dict':
        return d
    
    if format == "text":
        return LeftAligned()(d)
    
    if format == 'nx':
        import networkx as nx
        g = nx.DiGraph()
        def add_nodes(parent, d):
            for k in d.keys():
                g.add_edge(parent, k)
                add_nodes(k, d[k])
        add_nodes('hemibrain', d['hemibrain'])
        return g
        

@inject_client
def fetch_neurons(criteria, *, client=None):
    """
    Return properties and per-ROI synapse counts for a set of neurons.
    
    Searches for a set of Neurons (or Segments) that match the given :py:class:`.SegmentCriteria`.
    Returns their properties, including the distibution of their synapses in all brain regions.
    
    This implements a superset of the features on the Neuprint Explorer `Find Neurons`_ page.

    Returns data in the the same format as :py:func:`fetch_custom_neurons()`,
    but doesn't require you to write cypher.
    
    .. _Find Neurons: https://neuprint.janelia.org/?dataset=hemibrain%3Av1.0&qt=findneurons&q=1
    
    Args:
        criteria (:py:class:`.SegmentCriteria`):
            Only Neurons which satisfy all components of the given criteria are returned.

        client:
            If not provided, the global default :py:class:`.Client` will be used.
    
    Returns:
        Two DataFrames.
        ``(neurons_df, roi_counts_df)``
        
        In ``neurons_df``, all available ``:Neuron`` columns are returned, with the following changes:
        
            - ROI boolean columns are removed
            - ``roiInfo`` is parsed as json data
            - ``somaLocation`` is provided as a list ``[x, y, z]``
            - New columns ``input_rois`` and ``output_rois`` contain lists of each neuron's ROIs.
        
        In ``roi_counts_df``, the ``roiInfo`` has been loadded into a table
        of per-neuron-per-ROI synapse counts, with separate columns
        for ``pre`` (outputs) and ``post`` (inputs).

    See also:

        If you like the output format of this function but you want
        to provide your own cypher query, see :py:func:`.fetch_custom_neurons()`.


    Example:
    
        .. code-block:: ipython
        
            In [1]: from neuprint import fetch_neurons, SegmentCriteria as SC

            In [2]: neurons_df, roi_counts_df = fetch_neurons(
               ...:     SC(inputRois=['SIP(R)', 'aL(R)'],
               ...:        status='Traced',
               ...:        type='MBON.*',
               ...:        regex=True))
            
            In [3]: neurons_df.iloc[:5, :11]
            Out[3]:
                  bodyId                     instance    type cellBodyFiber   pre   post        size  status  cropped     statusLabel  somaRadius
            0  300972942                 MBON14(a3)_R  MBON14           NaN   543  13634  1563154937  Traced    False  Roughly traced         NaN
            1  422725634        MBON06(B1>a)(AVM07)_L  MBON06           NaN  1356  20978  3118269136  Traced    False  Roughly traced         NaN
            2  423382015        MBON23(a2sp)(PDL05)_R  MBON23          SFS1   733   4466   857093893  Traced    False  Roughly traced       291.0
            3  423774471       MBON19(a2p3p)(PDL05)_R  MBON19          SFS1   299   1484   628019179  Traced    False  Roughly traced       286.0
            4  424767514  MBON11(y1pedc>a/B)(ADM05)_R  MBON11        mAOTU2  1643  27641  5249327644  Traced    False          Traced       694.5
            
            In [4]: neurons_df['inputRois'].head()
            Out[4]:
            0    [MB(+ACA)(R), MB(R), None, SIP(R), SLP(R), SMP...
            1    [CRE(-ROB,-RUB)(R), CRE(R), INP, MB(+ACA)(R), ...
            2    [MB(+ACA)(R), MB(R), None, SIP(R), SLP(R), SMP...
            3    [MB(+ACA)(R), MB(R), SIP(R), SMP(R), SNP(R), a...
            4    [CRE(-ROB,-RUB)(R), CRE(L), CRE(R), INP, MB(+A...
            Name: inputRois, dtype: object
            
            In [5]: roi_counts_df.head()
            Out[5]:
                  bodyId          roi  pre   post
            0  300972942        MB(R)   17  13295
            1  300972942        aL(R)   17  13271
            2  300972942        a3(R)   17  13224
            3  300972942  MB(+ACA)(R)   17  13295
            4  300972942       SNP(R)  526    336
    """
    criteria = copy.copy(criteria)
    criteria.matchvar = 'n'

    # Unlike in fetch_custom_neurons() below, here we specify the
    # return properties individually to avoid a large JSON payload.
    # (Returning a map on every row is ~2x more costly than returning a table of rows/columns.)
    props = list(NEURON_COLS)
    props.remove('somaLocation')
    return_exprs = ',\n'.join(f'n.{prop} as {prop}' for prop in props)
    return_exprs = indent(return_exprs, ' '*15)[15:]
    
    q = f"""\
        MATCH (n :{criteria.label})
        {criteria.all_conditions(prefix=8)}
        RETURN {return_exprs},
               CASE
                 WHEN n.somaLocation IS NULL
                 THEN NULL ELSE [n.somaLocation.x, n.somaLocation.y, n.somaLocation.z]
               END as somaLocation
        ORDER BY n.bodyId
    """
    neuron_df = fetch_custom(q, client=client)
    neuron_df, roi_counts_df = _process_neuron_df(neuron_df, client)
    return neuron_df, roi_counts_df


@inject_client
def fetch_custom_neurons(q, *, client=None):
    """
    Return properties and per-ROI synapse counts for a set of neurons,
    using your own cypher query.
    
    Use a custom query to fetch a neuron table, with nicer output
    than you would get from a call to :py:func:`.fetch_custom()`.
    
    Returns data in the the same format as :py:func:`.fetch_neurons()`.
    but allows you to provide your own cypher query logic
    (subject to certain requirements; see below).

    This function includes all Neuron fields in the results,
    and also sends back ROI counts as a separate table.
    
    Args:

        q:
            Custom query. Must match a neuron named ``n``,
            and must ``RETURN n``.
            
            .. code-block::
            
                ...
                MATCH (n :Neuron)
                ...
                RETURN n
                ...

        neuprint_rois:
            Optional.  The list of ROI names from neuprint.
            If not provided, they will be fetched
            (so that ROI boolean columns can be dropped from the results).

        client:
            If not provided, the global default ``Client`` will be used.
        
    Returns:
        Two DataFrames.
        ``(neurons_df, roi_counts_df)``
        
        In ``neurons_df``, all available columns ``:Neuron`` columns are returned, with the following changes:
        
            - ROI boolean columns are removed
            - ``roiInfo`` is parsed as json data
            - ``somaLocation`` is provided as a list ``[x, y, z]``
            - New columns ``inputRoi`` and ``outputRoi`` contain lists of each neuron's ROIs.
        
        In ``roi_counts_df``, the ``roiInfo`` has been loadded into a table
        of per-neuron-per-ROI synapse counts, with separate columns
        for ``pre`` (outputs) and ``post`` (inputs).
    """
    results = client.fetch_custom(q)
    
    if len(results) == 0:
        neuron_df = pd.DataFrame([], columns=NEURON_COLS, dtype=object)
        roi_counts_df = pd.DataFrame([], columns=['bodyId', 'roi', 'pre', 'post'])
        return neuron_df, roi_counts_df
    
    neuron_df = pd.DataFrame(results['n'].tolist())

    # If somaLocation is already provided as a top-level column in the query results,
    # we assume the user's cypher query already converted it to [x,y,z] form.
    if 'somaLocation' in results:
        neuron_df['somaLocation'] = results['somaLocation']
    elif 'somaLocation' in neuron_df:
        no_soma = neuron_df['somaLocation'].isnull()
        neuron_df.loc[no_soma, 'somaLocation'] = None
        neuron_df.loc[~no_soma, 'somaLocation'] = neuron_df.loc[~no_soma, 'somaLocation'].apply(lambda sl: sl.get('coordinates'))

    neuron_df, roi_counts_df = _process_neuron_df(neuron_df, client)
    return neuron_df, roi_counts_df


def _process_neuron_df(neuron_df, client):
    """
    Given a DataFrame of neuron properties, parse the roiInfo into
    inputRois and outputRois, and a secondary DataFrame for per-ROI
    synapse counts.
    
    Returns:
        neuron_df, roi_counts_df
    
    Warning: destructively modifies the input DataFrame.
    """
    # Drop roi columns
    columns = {*neuron_df.columns} - {*client.all_rois}
    neuron_df = neuron_df[[*columns]]

    # Specify column order:
    # Standard columns first, than any extra columns in the results (if any).
    neuron_cols = [*filter(lambda c: c in neuron_df.columns, NEURON_COLS)]
    extra_cols = {*neuron_df.columns} - {*neuron_cols}
    neuron_cols += [*extra_cols]
    neuron_df = neuron_df[[*neuron_cols]]

    # Make a list of rois for every neuron (both pre and post)
    neuron_df['roiInfo'] = neuron_df['roiInfo'].apply(lambda s: ujson.loads(s))
    neuron_df['inputRois'] = neuron_df['roiInfo'].apply(lambda d: sorted([k for k,v in d.items() if v.get('post')]))
    neuron_df['outputRois'] = neuron_df['roiInfo'].apply(lambda d: sorted([k for k,v in d.items() if v.get('pre')]))

    # Return roi info as a separate table
    roi_counts = []
    for row in neuron_df.itertuples():
        for roi, counts in row.roiInfo.items():
            pre = counts.get('pre', 0)
            post = counts.get('post', 0)
            roi_counts.append( (row.bodyId, roi, pre, post) )

    roi_counts_df = pd.DataFrame(roi_counts, columns=['bodyId', 'roi', 'pre', 'post'])
    return neuron_df, roi_counts_df


@inject_client
@make_args_iterable(['rois'])
def fetch_simple_connections(upstream_criteria=None, downstream_criteria=None, rois=None, min_weight=1,
                             properties=['type', 'instance'],
                             *, client=None):
    """
    Find connections to/from small set(s) of neurons.
    
    Finds all connections from a set of "upstream" neurons,
    or to a set of "downstream" neurons,
    or all connections from a set of upstream neurons to a set of downstream neurons.

    Note:
        This function is not intended to be used with very large neuron sets.
        To fetch all adjacencies between a large set of neurons,
        set :py:func:`fetch_adjacencies()`, and additional ROI-filtering options.
        
        However, this function returns additional information on every row of the
        connection table, such as ``type`` and ``instance``, so it may be more
        convenient for small queries.

    Args:
        upstream_criteria:
            SegmentCriteria indicating how to filter for neurons
            on the presynaptic side of connections.
        downstream_criteria:
            SegmentCriteria indicating how to filter for neurons
            on the postsynaptic side of connections.
        rois:
            Limit results to neuron pairs that connect in at least one of the given ROIs.
        min_weight:
            Exclude connections whose total weight (across all ROIs) falls below this threshold.
        properties:
            Additional columns to include in the results, for both the upstream and downstream body.
        client:
            If not provided, the global default :py:class:`.Client` will be used.
    
    Returns:
        DataFrame
        One row per connection, with columns for upstream (pre-synaptic) and downstream (post-synaptic) properties.
    
    Example:
    
        .. code-block:: ipython
        
            In [1]: from neuprint import SegmentCriteria as SC, fetch_simple_connections
               ...: sources = [329566174, 425790257, 424379864, 329599710]
               ...: targets = [425790257, 424379864, 329566174, 329599710, 420274150]
               ...: fetch_simple_connections(SC(bodyId=sources), SC(bodyId=targets))
            Out[1]:
               bodyId_pre  bodyId_post  weight                   type_pre                  type_post         instance_pre        instance_post                                       conn_roiInfo
            0   329566174    425790257      43                    OA-VPM3                        APL   OA-VPM3(NO2/NO3)_R                APL_R  {'MB(R)': {'pre': 39, 'post': 39}, 'b'L(R)': {...
            1   329566174    424379864      37                    OA-VPM3                 AVM03e_pct   OA-VPM3(NO2/NO3)_R  AVM03e_pct(AVM03)_R  {'SNP(R)': {'pre': 34, 'post': 34}, 'SLP(R)': ...
            2   425790257    329566174      12                        APL                    OA-VPM3                APL_R   OA-VPM3(NO2/NO3)_R  {'MB(R)': {'pre': 12, 'post': 12}, 'gL(R)': {'...
            3   424379864    329566174       7                 AVM03e_pct                    OA-VPM3  AVM03e_pct(AVM03)_R   OA-VPM3(NO2/NO3)_R  {'SNP(R)': {'pre': 5, 'post': 5}, 'SLP(R)': {'...
            4   329599710    329566174       4  olfactory multi lvPN mALT                    OA-VPM3        mPNX(AVM06)_R   OA-VPM3(NO2/NO3)_R  {'SNP(R)': {'pre': 4, 'post': 4}, 'SIP(R)': {'...
            5   329566174    329599710       1                    OA-VPM3  olfactory multi lvPN mALT   OA-VPM3(NO2/NO3)_R        mPNX(AVM06)_R  {'SNP(R)': {'pre': 1, 'post': 1}, 'SLP(R)': {'...
            6   329566174    420274150       1                    OA-VPM3                 AVM03m_pct   OA-VPM3(NO2/NO3)_R  AVM03m_pct(AVM03)_R  {'SNP(R)': {'pre': 1, 'post': 1}, 'SLP(R)': {'...
    """
    SC = SegmentCriteria
    up_crit = copy.deepcopy(upstream_criteria)
    down_crit = copy.deepcopy(downstream_criteria)

    if up_crit is None:
        up_crit = SC(label='Neuron')
    if down_crit is None:
        down_crit = SC(label='Neuron')

    up_crit.matchvar = 'n'
    down_crit.matchvar = 'm'
    
    assert up_crit is not None or down_crit is not None, "No criteria specified"

    combined_conditions = SC.combined_conditions([up_crit, down_crit], ('n', 'm', 'e'), prefix=8)

    if min_weight > 1:
        weight_expr = dedent(f"""\
            WITH n, m, e
            WHERE e.weight >= {min_weight}
            """)
        weight_expr = indent(weight_expr, ' '*8)[8:] 
    else:
        weight_expr = ""

    rois = {*rois}
    if rois:
        invalid_rois = {*rois} - {*client.all_rois}
        assert not invalid_rois, f"Unrecognized ROIs: {invalid_rois}"

    return_props = ['n.bodyId as bodyId_pre',
                    'm.bodyId as bodyId_post',
                    'e.weight as weight']
    
    for p in properties:
        if p == 'roiInfo':
            return_props.append('apoc.convert.fromJsonMap(n.roiInfo) as roiInfo_pre')
            return_props.append('apoc.convert.fromJsonMap(m.roiInfo) as roiInfo_post')
        else:
            return_props.append(f'n.{p} as {p}_pre')
            return_props.append(f'm.{p} as {p}_post')
    
    return_props += ['e.roiInfo as conn_roiInfo']
    
    return_props_str = indent(',\n'.join(return_props), prefix=' '*15)[15:]

    q = f"""\
        MATCH (n:{up_crit.label})-[e:ConnectsTo]->(m:{down_crit.label})

        {combined_conditions}
        {weight_expr}
        RETURN {return_props_str}
        ORDER BY e.weight DESC,
                 n.bodyId,
                 m.bodyId
    """
    edges_df = client.fetch_custom(q)
    
    # Load connection roiInfo with ujson
    edges_df['conn_roiInfo'] = edges_df['conn_roiInfo'].apply(ujson.loads)
    
    if rois:
        keep = edges_df['conn_roiInfo'].apply(lambda roiInfo: bool(rois & {*roiInfo.keys()}))
        edges_df = edges_df.loc[keep].reset_index(drop=True)

    return edges_df


@inject_client
@make_args_iterable(['rois'])
def fetch_adjacencies(sources=None, targets=None, rois=None, min_roi_weight=1, min_total_weight=1, include_nonprimary=False, export_dir=None, batch_size=200, *, client=None):
    """
    Find connections to/from large set(s) of neurons, with per-ROI connection strengths.
    
    Fetch the adjacency table for connections amongst a set of neurons, broken down by ROI.
    Unless ``include_nonprimary=True``, only primary ROIs are included in the per-ROI connection table.
    Connections outside of the primary ROIs are labeled with the special name
    ``"NotPrimary"`` (which is not currently an ROI name in neuprint itself).

    Note:
        :py:func:`.fetch_simple_connections()` has similar functionality,
        but that function isn't suitable for querying large sets of neurons.
        It does, however, return additional information on every row of the 
        connection table, such as ``type`` and ``instance``, so it may be more
        convenient for small queries.

    Args:
        sources:
            Limit results to connections from bodies that match this criteria.
            Can be list of body IDs or :py:class:`.SegmentCriteria`. If ``None
            will include all bodies upstream of ``targets``.

        targets:
            Limit results to connections to bodies that match this criteria.
            Can be list of body IDs or :py:class:`.SegmentCriteria`. If ``None
            will include all bodies downstream of ``sources``.

        export_dir:
            Optional. Export CSV files for the neuron table,
            connection table (total weight), and connection table (per ROI).

        batch_size:
            For optimal performance, connections will be fetched in batches.
            This parameter specifies the batch size.

        rois:
            Limit results to connections within the listed ROIs.

        min_roi_weight:
            Limit results to connections of at least this strength within any particular ROI.
        
        min_total_weight:
            Limit results to connections that are at least this strong when totaled across all ROIs.
            
            Note:
                Even if ``min_roi_weight`` is also specified, all connections are counted towards satisfying
                the total weight threshold, even though some ROI entries are filtered out.
                Therefore, some connections in the results may appear not to satisfy ``min_total_weight``
                when their per-ROI weights are totaled.  That's just because you filtered out the weak
                per-ROI entries.

        include_nonprimary:
            If True, also list per-ROI totals for non-primary ROIs
            (i.e. parts of the ROI hierarchy that are sub-primary or super-primary).
            See :py:func:`fetch_roi_hierarchy` for details.

            Note:
                Since non-primary ROIs overlap with primary ROIs, then the sum of the
                ``weight`` column for each body pair will not be equal to the total
                connection strength between the bodies.
                (Some connections will be counted twice.)

        client:
            If not provided, the global default :py:class:`.Client` will be used.

    Returns:
        Two DataFrames, ``(traced_neurons_df, roi_conn_df)``, containing a
        table of neuron IDs and the per-ROI connection table, respectively.
        See caveat above concerning non-primary ROIs.

    See also:
        :py:func:`.fetch_simple_connections()`
        :py:func:`.fetch_traced_adjacencies()`

    Example:
    
        .. code-block:: ipython
        
            In [1]: from neuprint import Client
               ...: c = Client('neuprint.janelia.org', dataset='hemibrain:v1.0.1')
            
            In [2]: from neuprint import SegmentCriteria as SC, fetch_adjacencies
               ...: sources = [329566174, 425790257, 424379864, 329599710]
               ...: targets = [425790257, 424379864, 329566174, 329599710, 420274150]
               ...: neuron_df, connection_df = fetch_adjacencies(SC(bodyId=sources), SC(bodyId=targets))
            
            In [3]: neuron_df
            Out[3]:
                  bodyId             instance                       type
            0  329566174   OA-VPM3(NO2/NO3)_R                    OA-VPM3
            1  329599710        mPNX(AVM06)_R  olfactory multi lvPN mALT
            2  424379864  AVM03e_pct(AVM03)_R                 AVM03e_pct
            3  425790257                APL_R                        APL
            4  420274150  AVM03m_pct(AVM03)_R                 AVM03m_pct
            
            In [4]: connection_df
            Out[4]:
                bodyId_pre  bodyId_post     roi  weight
            0    329566174    329599710  SLP(R)       1
            1    329566174    420274150  SLP(R)       1
            2    329566174    424379864  SLP(R)      31
            3    329566174    424379864  SCL(R)       3
            4    329566174    424379864  SIP(R)       3
            5    329566174    425790257   gL(R)      17
            6    329566174    425790257   CA(R)      10
            7    329566174    425790257  CRE(R)       4
            8    329566174    425790257  b'L(R)       3
            9    329566174    425790257   aL(R)       3
            10   329566174    425790257  PED(R)       3
            11   329566174    425790257   bL(R)       2
            12   329566174    425790257  a'L(R)       1
            13   329599710    329566174  SLP(R)       3
            14   329599710    329566174  SIP(R)       1
            15   424379864    329566174  SLP(R)       4
            16   424379864    329566174  SCL(R)       2
            17   424379864    329566174  SIP(R)       1
            18   425790257    329566174   gL(R)       8
            19   425790257    329566174   CA(R)       3
            20   425790257    329566174   aL(R)       1

    **Total Connection Strength**

    To aggregate the per-ROI connection weights into total connection weights, use ``groupby(...)['weight'].sum()``
    
    .. code-block:: ipython

        In [5]: connection_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()
        Out[5]:
           bodyId_pre  bodyId_post  weight
        0   329566174    329599710       1
        1   329566174    420274150       1
        2   329566174    424379864      37
        3   329566174    425790257      43
        4   329599710    329566174       4
        5   424379864    329566174       7
        6   425790257    329566174      12
    """    
    rois = {*rois}
    invalid_rois = rois - {*client.all_rois}
    assert not invalid_rois, f"Unrecognized ROIs: {invalid_rois}"
    
    nonprimary_rois = rois - {*client.primary_rois}
    assert include_nonprimary or not nonprimary_rois, \
        f"Since you listed nonprimary rois ({nonprimary_rois}), please specify include_nonprimary=True"

    min_roi_weight = max(min_roi_weight, 1)
    min_total_weight = max(min_total_weight, min_roi_weight)

    def _prepare_criteria(criteria, matchvar):
        if criteria is None:
            criteria = SegmentCriteria(matchvar)

        # A previous version of fetch_adjacencies() accepted a list of bodyIds.
        # We still support that for now.
        if not isinstance(criteria, SegmentCriteria):
            assert isinstance(criteria, collections.abc.Iterable), \
                f"Invalid criteria: {criteria}"
            criteria = SegmentCriteria(matchvar, bodyId=criteria)

        criteria = copy.copy(criteria)
        criteria.matchvar = matchvar
        
        # If the user wants to filter for specific rois,
        # we can speed up the query by adding them to the SegmentCriteria
        if rois and not criteria.rois:
            criteria.rois = rois
            criteria.roi_req = 'any'

        return criteria

    def _fetch_neurons(criteria):
        matchvar = criteria.matchvar
        q = f"""\
            MATCH ({matchvar}:{criteria.label})
            {criteria.all_conditions(prefix=12)}
            RETURN {matchvar}.bodyId as bodyId,
                   {matchvar}.instance as instance,
                   {matchvar}.type as type
            ORDER BY {matchvar}.bodyId
        """
        return client.fetch_custom(q)

    # Ensure sources/targets are SegmentCriteria
    sources = _prepare_criteria(sources, 'n')
    targets = _prepare_criteria(targets, 'm')

    # Fetch neuron lists
    # (We'll need to filter these below, after we know
    # which ones are actually involved in connections.)
    sources_df = _fetch_neurons(sources)
    targets_df = _fetch_neurons(targets)

    # Concatenate sources and targets
    neurons_df = pd.concat((sources_df, targets_df), ignore_index=True)
    neurons_df.drop_duplicates('bodyId', inplace=True)    
    neurons_df.reset_index(drop=True, inplace=True)

    if not rois or min_total_weight <= 1:
        # If rois aren't specified, then we'll include 'NotPrimary' counts,
        # and that means we can't filter by weight in the query.
        # We'll filter afterwards.
        weight_condition = ""
    else:
        weight_condition = dedent(f"""\
            // -- Filter by total connection weight --
            WITH n,m,e
            WHERE e.weight >= {min_total_weight}
        """)
        weight_condition = indent(weight_condition, prefix=12*' ')[12:]
    
    # Fetch connections by batching either the source list
    # or the target list, not both.
    # (It turns out that batching across BOTH sources and
    # targets is much slower than batching across only one.)
    conn_tables = []
    
    if len(sources_df) <= len(targets_df):
        # Break sources into batches
        for batch_start in trange(0, len(sources_df), batch_size, disable=(len(sources_df) / batch_size < 1.0)):
            batch_stop = batch_start + batch_size
            source_bodies = sources_df['bodyId'].iloc[batch_start:batch_stop].tolist()
            
            batch_criteria = copy.copy(sources)
            batch_criteria.bodyId = source_bodies
            
            q = f"""\
                MATCH (n:{sources.label})-[e:ConnectsTo]->(m:{targets.label})
                {SegmentCriteria.combined_conditions((batch_criteria, targets), ('n', 'm', 'e'), prefix=12)}
                {weight_condition}
                RETURN n.bodyId as bodyId_pre,
                       m.bodyId as bodyId_post,
                       e.weight as weight,
                       e.roiInfo as roiInfo
            """
            conn_tables.append(client.fetch_custom(q))
    else:
        # Break targets into batches
        for batch_start in trange(0, len(targets_df), batch_size, disable=(len(targets_df) / batch_size < 1.0)):
            batch_stop = batch_start + batch_size
            target_bodies = targets_df['bodyId'].iloc[batch_start:batch_stop].tolist()
            
            batch_criteria = copy.copy(targets)
            batch_criteria.bodyId = target_bodies
            
            q = f"""\
                MATCH (n:{sources.label})-[e:ConnectsTo]->(m:{targets.label})
                {SegmentCriteria.combined_conditions((sources, batch_criteria), ('n', 'm', 'e'), prefix=12)}
                {weight_condition}
                RETURN n.bodyId as bodyId_pre,
                       m.bodyId as bodyId_post,
                       e.weight as weight,
                       e.roiInfo as roiInfo
            """
            conn_tables.append(client.fetch_custom(q))

    # Combine batches
    connections_df = pd.concat(conn_tables, ignore_index=True)

    # Parse roiInfo json (ujson is faster than apoc.convert.fromJsonMap)
    connections_df['roiInfo'] = connections_df['roiInfo'].apply(ujson.loads)

    # Extract per-ROI counts from the roiInfo column
    # to construct one big table of per-ROI counts
    roi_connections = []
    for row in connections_df.itertuples(index=False):
        # We use the 'post' count as the weight (ignore pre)
        roi_connections += [(row.bodyId_pre, row.bodyId_post, roi, weights.get('post', 0))
                            for roi, weights in row.roiInfo.items()]
    
    roi_conn_df = pd.DataFrame(roi_connections,
                               columns=['bodyId_pre', 'bodyId_post', 'roi', 'weight'])
    
    # Filter out non-primary ROIs
    primary_roi_conn_df = roi_conn_df.query('roi in @client.primary_rois')
    
    # Add a special roi name "NotPrimary" to account for the
    # difference between total weights and primary-only weights.
    primary_totals = primary_roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()

    totals_df = connections_df.merge(primary_totals, 'left', on=['bodyId_pre', 'bodyId_post'], suffixes=['_all', '_primary'])
    totals_df.fillna(0, inplace=True)
    totals_df['weight_notprimary'] = totals_df.eval('weight_all - weight_primary').astype(int)
    totals_df['roi'] = 'NotPrimary'
    
    # Drop weights other than NotPrimary
    totals_df = totals_df[['bodyId_pre', 'bodyId_post', 'roi', 'weight_notprimary']]
    notprimary_totals_df = totals_df.query('weight_notprimary > 0')
    notprimary_totals_df = notprimary_totals_df.rename(columns={'weight_notprimary': 'weight'})
    
    if not include_nonprimary:
        roi_conn_df = primary_roi_conn_df
    
    # Append NotPrimary rows to the connection table.
    roi_conn_df = pd.concat((roi_conn_df, notprimary_totals_df), ignore_index=True)
    roi_conn_df.sort_values(['bodyId_pre', 'bodyId_post', 'weight'], ascending=[True, True, False], inplace=True)
    roi_conn_df.reset_index(drop=True, inplace=True)
    
    # Consistency check: Double-check our math against the original totals
    summed_roi_weights = (roi_conn_df
                            .query('roi in @client.primary_rois or roi == "NotPrimary"')
                            .groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight']
                            .sum())
    compare_df = connections_df.merge(summed_roi_weights, 'left', on=['bodyId_pre', 'bodyId_post'], suffixes=['_orig', '_summed'])
    assert compare_df.fillna(0).eval('weight_orig == weight_summed').all()

    # Filter for the user's ROIs, if any
    if rois:
        roi_conn_df.query('roi in @rois and weight > 0', inplace=True)

    if min_total_weight > 1:
        total_weights_df = roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()
        keep_conns = total_weights_df.query('weight >= @min_total_weight')[['bodyId_pre', 'bodyId_post']]
        roi_conn_df = roi_conn_df.merge(keep_conns, 'inner', on=['bodyId_pre', 'bodyId_post'])

    # This is necessary, even if min_roi_weight == 1, to filter out zeros
    # that can occur in the case of weak inter-ROI connnections.
    roi_conn_df.query('weight >= @min_roi_weight', inplace=True)

    # Drop neurons that matched sources or targets but aren't mentioned in the final connection table.
    _connected_bodies = pd.unique(roi_conn_df[['bodyId_pre', 'bodyId_post']].values.reshape(-1))
    neurons_df.query('bodyId in @_connected_bodies', inplace=True)
    neurons_df.reset_index(drop=True, inplace=True)
    
    # Export to CSV
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

        # Export Nodes
        p = f"{export_dir}/neurons.csv"
        neurons_df.to_csv(p, index=False, header=True)
        
        # Export Edges (per ROI)
        p = f"{export_dir}/roi-connections.csv"
        roi_conn_df.to_csv(p, index=False, header=True)

        # Export Edges (total weight)
        p = f"{export_dir}/total-connections.csv"
        connections_df[['bodyId_pre', 'bodyId_post', 'weight']].to_csv(p, index=False, header=True)

    return neurons_df, roi_conn_df


@inject_client
def fetch_traced_adjacencies(export_dir=None, batch_size=200, *, client=None):
    """
    Convenience function that calls :py:func:`.fetch_adjacencies()`
    for all ``Traced``, non-``cropped`` neurons. 
 
    Note:
        On the hemibrain dataset, this function takes a few minutes to run,
        and the results are somewhat large (~300 MB).
    
    Example:
        
        .. code-block:: ipython
        
            In [1]: neurons_df, roi_conn_df = fetch_traced_adjacencies('exported-connections')

            In [2]: roi_conn_df.head()
            Out[2]:
                   bodyId_pre  bodyId_post        roi  weight
            0      5813009352    516098538     SNP(R)       2
            1      5813009352    516098538     SLP(R)       2
            2       326119769    516098538     SNP(R)       1
            3       326119769    516098538     SLP(R)       1
            4       915960391    202916528         FB       1

            In [3]: # Obtain total weights (instead of per-connection-per-ROI weights)
               ...: conn_groups = roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)
               ...: total_conn_df = conn_groups['weight'].sum()
               ...: total_conn_df.head()
            Out[3]:
               bodyId_pre  bodyId_post  weight
            0   202916528    203253253       2
            1   202916528    203257652       2
            2   202916528    203598557       2
            3   202916528    234292899       4
            4   202916528    264986706       2        
     """
    criteria = SegmentCriteria(status="Traced", cropped=False)
    return fetch_adjacencies(criteria, criteria, include_nonprimary=False, export_dir=export_dir, batch_size=batch_size, client=client)


@inject_client
def fetch_common_connectivity(criteria, search_direction='upstream', min_weight=1, properties=['type', 'instance'], *, client=None):
    """
    Find shared connections among a set of neurons.
    
    Given a set of neurons that match the given criteria, find neurons
    that connect to ALL of the neurons in the set, i.e. connections
    that are common to all neurons in the matched set.
    
    This is the Python equivalent to the Neuprint Explorer `Common Connectivity`_ page.

    .. _Common Connectivity: https://neuprint.janelia.org/?dataset=hemibrain%3Av1.0&qt=commonconnectivity&q=1
    
    
    Args:
        criteria:
            :py:class:`.SegmentCriteria` used to determine the match set,
            for which common connections will be found.

        search_direction (``"upstream"`` or ``"downstream"``):
            Whether or not to search for common connections upstream of
            the matched neurons or downstream of the matched neurons. 
        
        min_weight:
            Connections below the given strength will not be included in the results.

        properties:
            Additional columns to include in the results, for both the upstream and downstream body.

        client:
            If not provided, the global default :py:class:`.Client` will be used.
    
    Returns: DataFrame
        (Same format as returned by :py:func:`fetch_simple_connections()`.)
        One row per connection, with columns for upstream and downstream properties.
        For instance, if ``search_direction="upstream"``, then the matched neurons will appear
        in the ``downstream_`` columns, and the common connections will appear in the ``upstream_``
        columns.
    """
    assert search_direction in ('upstream', 'downstream')
    if search_direction == "upstream":
        edges_df = fetch_simple_connections(None, criteria, min_weight, properties, client=client)
        
        # How bodies many met main search criteria?
        num_primary = edges_df['bodyId_post'].nunique()

        # upstream bodies that connect to ALL of the main bodies are the 'common' bodies.
        upstream_counts = edges_df['bodyId_pre'].value_counts()
        _keep = upstream_counts[upstream_counts == num_primary].index
        return edges_df.query('bodyId_pre in @_keep')

    if search_direction == "downstream":
        edges_df = fetch_simple_connections(criteria, None, min_weight, properties, client=client)
        
        # How bodies many met main search criteria?
        num_primary = edges_df['bodyId_pre'].nunique()

        # upstream bodies that connect to ALL of the main are the 'common' bodies.
        upstream_counts = edges_df['bodyId_post'].value_counts()
        _keep = upstream_counts[upstream_counts == num_primary].index
        return edges_df.query('bodyId_post in @_keep')


@inject_client
def fetch_shortest_paths(upstream_bodyId, downstream_bodyId, min_weight=1,
                         intermediate_criteria=None,
                         timeout=5.0, *, client=None):
    """
    Find all neurons along the shortest path between two neurons.
    
    Args:
        upstream_bodyId:
            The starting neuron
        
        downstream_bodyId:
            The destination neuron
        
        min_weight:
            Minimum connection strength for each step in the path.
        
        intermediate_criteria (:py:class:`.SegmentCriteria`):
            Filtering criteria for neurons on path.
            All intermediate neurons in the path must satisfy this criteria.
            By default, ``SegmentCriteria(status="Traced")`` is used.
        
        timeout:
            Give up after this many seconds, in which case an **empty DataFrame is returned.**
            No exception is raised!
        
        client:
            If not provided, the global default :py:class:`.Client` will be used.
    
    Returns:
        All paths are concatenated into a single DataFrame.
        The `path` column indicates which path that row belongs to.
        The `weight` column indicates the connection strength to that
        body from the previous body in the path.
        
    Example:
    
        .. code-block:: ipython
        
            In [1]: fetch_shortest_paths(329566174, 294792184, min_weight=10)
            Out[1]:
                  path     bodyId                       type  weight
            0        0  329566174                    OA-VPM3       0
            1        0  517169460                 PDL05h_pct      11
            2        0  297251714                ADM01om_pct      15
            3        0  294424196                PDL13ob_pct      11
            4        0  295133927               PDM18a_d_pct      10
            ...    ...        ...                        ...     ...
            5773   962  511271574                 ADL24h_pct      43
            5774   962  480923210                PDL10od_pct      18
            5775   962  294424196                PDL13ob_pct      21
            5776   962  295133927               PDM18a_d_pct      10
            5777   962  294792184  olfactory multi vPN mlALT      10
            
            [5778 rows x 4 columns]
    """
    if intermediate_criteria is None:
        intermediate_criteria = SegmentCriteria(status="Traced")
    
    assert len(intermediate_criteria.inputRois) == 0 and len(intermediate_criteria.outputRois) == 0, \
        "This function doesn't support search criteria that specifies inputRois or outputRois. "\
        "You can specify generic (intersecting) rois, though."

    intermediate_criteria = copy.copy(intermediate_criteria)
    intermediate_criteria.matchvar = 'n'

    timeout_ms = int(1000*timeout)

    nodes_where = intermediate_criteria.basic_conditions(comments=False)
    nodes_where += f"\n OR n.bodyId in [{upstream_bodyId}, {downstream_bodyId}]"
    nodes_where = nodes_where.replace('\n', '')
    
    q = f"""\
        call apoc.cypher.runTimeboxed(
            "MATCH (src :Neuron {{ bodyId: {upstream_bodyId} }}),
                   (dest:Neuron {{ bodyId: {downstream_bodyId} }}),
                   p = allShortestPaths((src)-[:ConnectsTo*]->(dest))

            WHERE     ALL (x in relationships(p) WHERE x.weight >= {min_weight})
                  AND ALL (n in nodes(p) {nodes_where})

            RETURN [n in nodes(p) | [n.bodyId, n.type]] AS path,
                   [x in relationships(p) | x.weight] AS weights",
            
            {{}},{timeout_ms}) YIELD value
            RETURN value.path as path, value.weights AS weights
    """
    results_df = fetch_custom(q)
    
    table_indexes = []
    table_bodies = []
    table_types = []
    table_weights = []

    for path_index, (path, weights) in enumerate(results_df.itertuples(index=False)):
        bodies, types = zip(*path)
        weights = [0, *weights]
        
        table_indexes += len(bodies)*[path_index]
        table_bodies += bodies
        table_types += types
        table_weights += weights

    paths_df = pd.DataFrame({'path': table_indexes,
                             'bodyId': table_bodies,
                             'type': table_types,
                             'weight': table_weights})
    return paths_df


@inject_client
def fetch_synapses(segment_criteria, synapse_criteria=None, *, client=None):
    """
    Fetch synapses from a neuron or selection of neurons.

    Args:
    
        segment_criteria (SegmentCriteria or bodyId list):
            Can be either a single bodyID, a list-like of multiple body IDs,
            a DataFrame with a `bodyId` column or a :py:class:`.SegmentCriteria`
            used to find a set of body IDs.

            Note:
                Any ROI criteria specified in this argument does not affect
                which synapses are returned, only which bodies are inspected.

        synapse_criteria (SynapseCriteria):
            Optional. Allows you to filter synapses by roi, type, confidence.
            See :py:class:`.SynapseCriteria` for details.

            If the criteria specifies ``primary_only=True`` only primary ROIs will be returned in the results.
            If a synapse does not intersect any primary ROI, it will be listed with an roi of ``None``.
            (Since 'primary' ROIs do not overlap, each synapse will be listed only once.)
            Otherwise, all ROI names will be included in the results.
            In that case, some synapses will be listed multiple times -- once per intersecting ROI.
            If a synapse does not intersect any ROI, it will be listed with an roi of ``None``.

        client:
            If not provided, the global default :py:class:`.Client` will be used.

    Returns:
    
        DataFrame in which each row represent a single synapse.
        Unless ``primary_only`` was specified, some synapses may be listed more than once,
        if they reside in more than one overlapping ROI.
    
    Example:
    
        .. code-block:: ipython
        
            In [1]: from neuprint import SegmentCriteria as SC, SynapseCriteria as SynC, fetch_synapses
               ...: fetch_synapses(SC(type='ADL.*', regex=True, rois=['FB']),
               ...:                SynC(rois=['LH(R)', 'SIP(R)'], primary_only=True))
            Out[1]:
                    bodyId  type     roi      x      y      z  confidence
            0   5812983094   pre  SIP(R)  15300  25268  14043    0.992000
            1   5812983094   pre  SIP(R)  15300  25268  14043    0.992000
            2   5812983094   pre  SIP(R)  15300  25268  14043    0.992000
            3   5812983094   pre  SIP(R)  15300  25268  14043    0.992000
            4   5812983094   pre   LH(R)   5447  21281  19155    0.991000
            5   5812983094   pre   LH(R)   5434  21270  19201    0.995000
            6   5812983094   pre   LH(R)   5434  21270  19201    0.995000
            7   5812983094   pre   LH(R)   5434  21270  19201    0.995000
            8   5812983094   pre   LH(R)   5447  21281  19155    0.991000
            9   5812983094   pre   LH(R)   5447  21281  19155    0.991000
            10  5812983094   pre   LH(R)   5447  21281  19155    0.991000
            11  5812983094   pre   LH(R)   5434  21270  19201    0.995000
            12  5812983094   pre   LH(R)   5447  21281  19155    0.991000
            13  5812983094   pre   LH(R)   5447  21281  19155    0.991000
            14  5812983094   pre   LH(R)   5447  21281  19155    0.991000
            15  5812983094   pre   LH(R)   5447  21281  19155    0.991000
            16  5812983094   pre   LH(R)   5447  21281  19155    0.991000
            17  5812983094   pre   LH(R)   5434  21270  19201    0.995000
            18  5812983094   pre   LH(R)   5447  21281  19155    0.991000
            19  5812983094   pre   LH(R)   5434  21270  19201    0.995000
            20  5812983094   pre   LH(R)   5447  21281  19155    0.991000
            21  5812983094   pre   LH(R)   5434  21270  19201    0.995000
            22  5812983094   pre   LH(R)   5447  21281  19155    0.991000
            23  5812983094   pre  SIP(R)  15300  25268  14043    0.992000
            24  5812983094   pre  SIP(R)  15300  25268  14043    0.992000
            25  5812983094   pre  SIP(R)  15300  25268  14043    0.992000
            26  5812983094   pre  SIP(R)  15300  25268  14043    0.992000
            27  5812983094   pre  SIP(R)  15300  25268  14043    0.992000
            28  5812983094   pre  SIP(R)  15300  25268  14043    0.992000
            29   764377593  post   LH(R)   8043  21587  15146    0.804000
            30   764377593  post   LH(R)   8057  21493  15140    0.906482
            31   859152522   pre  SIP(R)  13275  25499  13629    0.997000
            32   859152522   pre  SIP(R)  13275  25499  13629    0.997000
            33   859152522  post  SIP(R)  13349  25337  13653    0.818386
            34   859152522  post  SIP(R)  12793  26362  14202    0.926918
            35   859152522   pre  SIP(R)  13275  25499  13629    0.997000

    """
    if isinstance(segment_criteria, pd.DataFrame):
        assert 'bodyId' in segment_criteria.columns, \
            'If passing a DataFrame, it must have "bodyId" column'
        segment_criteria = SegmentCriteria(bodyId=segment_criteria['bodyId'].values, client=client)
    elif not isinstance(segment_criteria, SegmentCriteria):
        segment_criteria = SegmentCriteria(bodyId=segment_criteria, client=client)

    assert isinstance(segment_criteria, SegmentCriteria), \
        ("Please pass a SegmentCriteria, a list of bodyIds, "
         f"or a DataFrame with a 'bodyId' column, not {segment_criteria}")

    segment_criteria = copy.copy(segment_criteria)
    segment_criteria.matchvar = 'n'
    
    if synapse_criteria is None:
        synapse_criteria = SynapseCriteria()

    if synapse_criteria.primary_only:
        return_rois = {*client.primary_rois}
    else:
        return_rois = {*client.all_rois}

    # If the user specified rois to filter synapses by, but hasn't specified rois
    # in the SegmentCriteria, add them to the SegmentCriteria to speed up the query.
    if synapse_criteria.rois and not segment_criteria.rois:
        segment_criteria.rois = {*synapse_criteria.rois}
        segment_criteria.roi_req = 'any'

    # Fetch results
    cypher = dedent(f"""\
        MATCH (n:{segment_criteria.label})
        {segment_criteria.all_conditions('n', prefix=8)}

        MATCH (n)-[:Contains]->(ss:SynapseSet),
              (ss)-[:Contains]->(s:Synapse)

        {synapse_criteria.condition('n', 's', prefix=8)}

        // De-duplicate 's' because 'pre' synapses can appear in more than one SynapseSet
        WITH DISTINCT n, s
        
        RETURN n.bodyId as bodyId,
               s.type as type,
               s.confidence as confidence,
               s.location.x as x,
               s.location.y as y,
               s.location.z as z,
               apoc.map.removeKeys(s, ['location', 'confidence', 'type']) as syn_info
    """)
    data = client.fetch_custom(cypher, format='json')['data']

    # Assemble DataFrame
    syn_table = []
    for body, syn_type, conf, x, y, z, syn_info in data:
        # Exclude non-primary ROIs if necessary
        syn_rois = return_rois & {*syn_info.keys()}
        for roi in syn_rois:
            syn_table.append((body, syn_type, roi, x, y, z, conf))

        if not syn_rois:
            syn_table.append((body, syn_type, None, x, y, z, conf))

    syn_df = pd.DataFrame(syn_table, columns=['bodyId', 'type', 'roi', 'x', 'y', 'z', 'confidence'])

    # Save RAM with smaller dtypes and interned strings
    syn_df['type'] = pd.Categorical(syn_df['type'], ['pre', 'post'])
    syn_df['roi'] = syn_df['roi'].apply(lambda s: sys.intern(s) if s else s)
    syn_df['x'] = syn_df['x'].astype(np.int32)
    syn_df['y'] = syn_df['y'].astype(np.int32)
    syn_df['z'] = syn_df['z'].astype(np.int32)
    syn_df['confidence'] = syn_df['confidence'].astype(np.float32)

    return syn_df


@inject_client
def fetch_synapse_connections(source_criteria=None, target_criteria=None, synapse_criteria=None, *, client=None):
    """
    Fetch synaptic-level connections between source and target neurons.
    
    Note:
        Use this function if you need information about individual synapse connections,
        such as their exact positions or confidence scores.
        If you're just interested in aggregate neuron-to-neuron connection info
        (including connection strengths and ROI intersections), see
        :py:func:`fetch_simple_connections()` and :py:func:`fetch_adjacencies()`,
        which are faster and have more condensed outputs than this function.
    
    Args:
    
        source_criteria (SegmentCriteria or bodyId list):
            Criteria to by which to filter source (pre-synaptic) neurons.
            If omitted, all Neurons will be considered as possible sources.

            Note:
                Any ROI criteria specified in this argument does not affect
                which synapses are returned, only which bodies are inspected.

        target_criteria (SegmentCriteria or bodyId list):
            Criteria to by which to filter target (post-synaptic) neurons.
            If omitted, all Neurons will be considered as possible sources.

            Note:
                Any ROI criteria specified in this argument does not affect
                which synapses are returned, only which bodies are inspected.

        synapse_criteria (SynapseCriteria):
            Optional. Allows you to filter synapses by roi, type, confidence.
            The same criteria is used to filter both ``pre`` and ``post`` sides
            of the connection.
            By default, ``SynapseCriteria(primary_only=True)`` is used.
            
            If ``primary_only`` is specified in the criteria, then the resulting
            ``roi_pre`` and ``roi_post`` columns will contain a single
            string (or ``None``) in every row.
            
            Otherwise, the roi columns will contain a list of ROIs for every row.
            (Primary ROIs do not overlap, so every synapse resides in only one
            (or zero) primary ROI.)
            See :py:class:`.SynapseCriteria` for details.

        client:
            If not provided, the global default :py:class:`.Client` will be used.

    Returns:
    
        DataFrame in which each row represents a single synaptic connection
        between an upstream (pre-synaptic) body and downstream (post-synaptic) body.

        Synapse locations are listed in columns ``[x_pre, y_pre, z_pre]`` and
        ``[x_post, y_post, z_post]`` for the upstream and downstream synapses,
        respectively.

        The ``roi_pre`` and ``roi_post`` columns will contain either strings
        or lists-of-strings, depending on the ``primary_only`` synapse criteria as
        described above.
    
    Example:

        .. code-block:: ipython

            In [1]: from neuprint import fetch_synapse_connections, SegmentCriteria as SC, SynapseCriteria as SynC
               ...: fetch_synapse_connections(SC(bodyId=792368888), None, SynC(rois=['PED(R)', 'SMP(R)'], primary_only=True))
            Out[1]:
                bodyId_pre  bodyId_post roi_pre roi_post  x_pre  y_pre  z_pre  x_post  y_post  z_post  confidence_pre  confidence_post
            0    792368888    754547386  PED(R)   PED(R)  14013  27747  19307   13992   27720   19313           0.996         0.401035
            1    792368888    612742248  PED(R)   PED(R)  14049  27681  19417   14044   27662   19408           0.921         0.881487
            2    792368888   5901225361  PED(R)   PED(R)  14049  27681  19417   14055   27653   19420           0.921         0.436177
            3    792368888   5813117385  SMP(R)   SMP(R)  23630  29443  16297   23634   29437   16279           0.984         0.970746
            4    792368888   5813083733  SMP(R)   SMP(R)  23630  29443  16297   23634   29419   16288           0.984         0.933871
            5    792368888   5813058320  SMP(R)   SMP(R)  18662  34144  12692   18655   34155   12697           0.853         0.995000
            6    792368888   5812981989  PED(R)   PED(R)  14331  27921  20099   14351   27928   20085           0.904         0.877373
            7    792368888   5812981381  PED(R)   PED(R)  14331  27921  20099   14301   27919   20109           0.904         0.567321
            8    792368888   5812981381  PED(R)   PED(R)  14013  27747  19307   14020   27747   19285           0.996         0.697836
            9    792368888   5812979314  PED(R)   PED(R)  14331  27921  20099   14329   27942   20109           0.904         0.638362
            10   792368888    424767514  PED(R)   PED(R)  14331  27921  20099   14324   27934   20085           0.904         0.985734
            11   792368888    424767514  PED(R)   PED(R)  14013  27747  19307   14020   27760   19294           0.996         0.942831
            12   792368888    424767514  PED(R)   PED(R)  14049  27681  19417   14040   27663   19420           0.921         0.993586
            13   792368888    331662710  SMP(R)   SMP(R)  23630  29443  16297   23644   29429   16302           0.984         0.996389
            14   792368888   1196854070  PED(R)   PED(R)  14331  27921  20099   14317   27935   20101           0.904         0.968408
            15   792368888   1131831702  SMP(R)   SMP(R)  23630  29443  16297   23651   29434   16316           0.984         0.362952
    """
    def prepare_sc(sc, matchvar):
        if sc is None:
            sc = SegmentCriteria()
        
        if isinstance(sc, pd.DataFrame):
            assert 'bodyId' in sc.columns, \
                'If passing a DataFrame, it must have "bodyId" column'
            sc = SegmentCriteria(bodyId=sc['bodyId'].values, client=client)
        elif not isinstance(sc, SegmentCriteria):
            sc = SegmentCriteria(bodyId=sc, client=client)
    
        assert isinstance(sc, SegmentCriteria), \
            ("Please pass a SegmentCriteria, a list of bodyIds, "
             f"or a DataFrame with a 'bodyId' column, not {sc}")
    
        sc = copy.copy(sc)
        sc.matchvar = matchvar
        
        # If the user specified rois to filter synapses by, but hasn't specified rois
        # in the SegmentCriteria, add them to the SegmentCriteria to speed up the query.
        if sc.rois and not sc.rois:
            sc.rois = {*synapse_criteria.rois}
            sc.roi_req = 'any'
    
        return sc

    assert source_criteria is not None or target_criteria is not None, \
        "Please specify either source or target search criteria (or both)."

    source_criteria = prepare_sc(source_criteria, 'n')
    target_criteria = prepare_sc(target_criteria, 'm')

    if synapse_criteria is None:
        synapse_criteria = SynapseCriteria(primary_only=True)
    
    if synapse_criteria.primary_only:
        return_rois = {*client.primary_rois}
    else:
        return_rois = {*client.all_rois}

    source_syn_crit = copy.copy(synapse_criteria)
    target_syn_crit = copy.copy(synapse_criteria)

    source_syn_crit.matchvar = 'ns'
    target_syn_crit.matchvar = 'ms'

    source_syn_crit.type = 'pre'
    target_syn_crit.type = 'post'
    
    # Fetch results
    cypher = dedent(f"""\
        MATCH (n:{source_criteria.label})-[:ConnectsTo]->(m:{target_criteria.label}),
              (n)-[:Contains]->(nss:SynapseSet),
              (m)-[:Contains]->(mss:SynapseSet),
              (nss)-[:ConnectsTo]->(mss),
              (nss)-[:Contains]->(ns:Synapse),
              (mss)-[:Contains]->(ms:Synapse),
              (ns)-[:SynapsesTo]->(ms)

        {SegmentCriteria.combined_conditions((source_criteria, target_criteria), ('n', 'm', 'ns', 'ms'), prefix=8)}

        {source_syn_crit.condition('n', 'm', 'ns', 'ms', prefix=8)}
        {target_syn_crit.condition('n', 'm', 'ns', 'ms', prefix=8)}
        
        RETURN n.bodyId as bodyId_pre,
               m.bodyId as bodyId_post,
               ns.location.x as ux,
               ns.location.y as uy,
               ns.location.z as uz,
               ms.location.x as dx,
               ms.location.y as dy,
               ms.location.z as dz,
               ns.confidence as confidence_pre,
               ms.confidence as confidence_post,
               apoc.map.removeKeys(ns, ['location', 'confidence', 'type']) as info_pre,
               apoc.map.removeKeys(ms, ['location', 'confidence', 'type']) as info_post
    """)
    data = client.fetch_custom(cypher, format='json')['data']

    # Assemble DataFrame
    syn_table = []
    for bodyId_pre, bodyId_post, ux, uy, uz, dx, dy, dz, up_conf, dn_conf, info_pre, info_post in data:
        # Exclude non-primary ROIs if necessary
        pre_rois = return_rois & {*info_pre.keys()}
        post_rois = return_rois & {*info_post.keys()}

        # Intern the ROIs to save RAM
        pre_rois = sorted(map(sys.intern, pre_rois))
        post_rois = sorted(map(sys.intern, post_rois))

        pre_rois = pre_rois or [None]
        post_rois = post_rois or [None]

        # Should be (at most) one ROI when primary_only=True,
        # so only show that one (not a list)
        if synapse_criteria.primary_only:
            pre_rois = pre_rois[0]
            post_rois = post_rois[0]
        
        syn_table.append((bodyId_pre, bodyId_post, pre_rois, post_rois, ux, uy, uz, dx, dy, dz, up_conf, dn_conf))

    syn_df = pd.DataFrame(syn_table, columns=['bodyId_pre', 'bodyId_post',
                                              'roi_pre', 'roi_post',
                                              'x_pre', 'y_pre', 'z_pre', 'x_post', 'y_post', 'z_post',
                                              'confidence_pre', 'confidence_post'])

    # Save RAM with smaller dtypes
    syn_df['x_pre'] = syn_df['x_pre'].astype(np.int32)
    syn_df['y_pre'] = syn_df['y_pre'].astype(np.int32)
    syn_df['z_pre'] = syn_df['z_pre'].astype(np.int32)
    syn_df['x_post'] = syn_df['x_post'].astype(np.int32)
    syn_df['y_post'] = syn_df['y_post'].astype(np.int32)
    syn_df['z_post'] = syn_df['z_post'].astype(np.int32)
    syn_df['confidence_pre'] = syn_df['confidence_pre'].astype(np.float32)
    syn_df['confidence_post'] = syn_df['confidence_post'].astype(np.float32)

    return syn_df
