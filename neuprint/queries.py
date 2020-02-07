import os
import sys
import copy
import collections
from textwrap import indent, dedent

import numpy as np
import pandas as pd
from tqdm import trange

from .client import inject_client
from .segmentcriteria import SegmentCriteria
from .synapsecriteria import SynapseCriteria
from .utils import make_args_iterable

# ujson is faster than Python's builtin json module
import ujson


@inject_client
def fetch_custom(cypher, dataset="", format='pandas', *, client=None):
    """
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
def fetch_neurons(criteria, *, client=None):
    """
    Search for a set of Neurons (or Segments) that match the given :py:class:`.SegmentCriteria`.
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

    q = f"""\
        MATCH (n :{criteria.label})
        {criteria.all_conditions(prefix=8)}
        RETURN n
        ORDER BY n.bodyId
    """
    
    return fetch_custom_neurons(q, client=client)



@inject_client
def fetch_custom_neurons(q, *, client=None):
    """
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
    
    neuron_cols = ['bodyId', 'instance', 'type', 'cellBodyFiber',
                   'pre', 'post', 'size',
                   'status', 'cropped', 'statusLabel',
                   'somaRadius', 'somaLocation',
                   'inputRois', 'outputRois', 'roiInfo']
    
    if len(results) == 0:
        neuron_df = pd.DataFrame([], columns=neuron_cols, dtype=object)
        roi_counts_df = pd.DataFrame([], columns=['bodyId', 'roi', 'pre', 'post'])
        return neuron_df, roi_counts_df
    
    neuron_df = pd.DataFrame(results['n'].tolist())
    
    # Drop roi columns
    columns = {*neuron_df.columns} - {*client.all_rois}
    neuron_df = neuron_df[[*columns]]

    # Extract somaLocation
    if 'somaLocation' in neuron_df:
        no_soma = neuron_df['somaLocation'].isnull()
        neuron_df.loc[no_soma, 'somaLocation'] = None
        neuron_df.loc[~no_soma, 'somaLocation'] = neuron_df.loc[~no_soma, 'somaLocation'].apply(lambda sl: sl.get('coordinates'))
    
    # Specify column order:
    # Standard columns first, than any extra columns in the results (if any).
    neuron_cols = [*filter(lambda c: c in neuron_df.columns, neuron_cols)]
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
def fetch_simple_connections(upstream_criteria=None, downstream_criteria=None, min_weight=1,
                             properties=['type', 'instance'],
                             *, client=None):
    """
    Find all connections from a set of "upstream" neurons,
    or to a set of "downstream" neurons,
    or all connections from a set of upstream neurons to a set of downstream neurons.

    Args:
        upstream_criteria:
            SegmentCriteria indicating how to filter for neurons
            on the presynaptic side of connections.
        downstream_criteria:
            SegmentCriteria indicating how to filter for neurons
            on the postsynaptic side of connections.
        min_weight:
            Exclude connections below this weight.
        properties:
            Additional columns to include in the results, for both the upstream and downstream body.
        client:
            If not provided, the global default :py:class:`.Client` will be used.
    
    Returns:
        DataFrame
        One row per connection, with columns for upstream and downstream properties.
        
    Note:
        This function is not intended to be used with very large neuron sets.
        To fetch all adjacencies between a large set of neurons,
        set :py:func:`fetch_adjacencies()`, which queries the server in batches.
    """
    SC = SegmentCriteria
    up_crit = copy.deepcopy(upstream_criteria)
    down_crit = copy.deepcopy(downstream_criteria)

    if up_crit is None:
        up_crit = SC(label='Neuron')
    if down_crit is None:
        down_crit = SC(label='Neuron')

    up_crit.matchvar = 'upstream'
    down_crit.matchvar = 'downstream'
    
    assert up_crit is not None or down_crit is not None, "No criteria specified"

    combined_conditions = SC.combined_conditions([up_crit, down_crit],
                                                 ('upstream', 'downstream', 'e'),
                                                 prefix=8)

    if min_weight > 1:
        weight_expr = dedent(f"""\
            WITH upstream, downstream, e
            WHERE e.weight >= {min_weight}\
            """)
        weight_expr = indent(weight_expr, ' '*8)[8:] 
    else:
        weight_expr = ""

    return_props = ['upstream.bodyId', 'downstream.bodyId', 'e.weight as weight']
    if properties:
        return_props += [f'upstream.{p}' for p in properties]
        return_props += [f'downstream.{p}' for p in properties]
    
    return_props_str = indent(',\n'.join(return_props), prefix=' '*15)[15:]

    # If roiInfo is requested, convert from json
    return_props_str = return_props_str.replace('upstream.roiInfo',
                            'apoc.convert.fromJsonMap(upstream.roiInfo) as upstream_roiInfo')
    return_props_str = return_props_str.replace('downstream.roiInfo',
                            'apoc.convert.fromJsonMap(downstream.roiInfo) as downstream_roiInfo')

    q = f"""\
        MATCH (upstream:{up_crit.label})-[e:ConnectsTo]->(downstream:{down_crit.label})

        {combined_conditions}
        {weight_expr}
        RETURN {return_props_str}
        ORDER BY e.weight DESC,
                 upstream.bodyId,
                 downstream.bodyId
    """
    edges_df = client.fetch_custom(q)
    
    # Rename columns: Replace '.' with '_'.
    renames = {col: col.replace('.', '_') for col in edges_df.columns}
    edges_df.rename(columns=renames, inplace=True)
    return edges_df


@inject_client
def fetch_common_connectivity(criteria, search_direction='upstream', min_weight=1, properties=['type', 'instance'], *, client=None):
    """
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
        
        # How bodies many met primary serach criteria?
        num_primary = edges_df['downstream_bodyId'].nunique()

        # upstream bodies that connect to ALL of the primary are the 'common' bodies.
        upstream_counts = edges_df['upstream_bodyId'].value_counts()
        keep = upstream_counts[upstream_counts == num_primary].index
        
        return edges_df.query('upstream_bodyId in @keep')

    if search_direction == "downstream":
        edges_df = fetch_simple_connections(criteria, None, min_weight, properties, client=client)
        
        # How bodies many met primary serach criteria?
        num_primary = edges_df['upstream_bodyId'].nunique()

        # upstream bodies that connect to ALL of the primary are the 'common' bodies.
        upstream_counts = edges_df['downstream_bodyId'].value_counts()
        keep = upstream_counts[upstream_counts == num_primary].index
        
        return edges_df.query('downstream_bodyId in @keep')


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
def fetch_adjacencies(sources=None, targets=None, export_dir=None, batch_size=200, *, client=None):
    """
    Fetch the adjacency table for connections amongst a set of neurons, broken down by ROI.
    Only primary ROIs are included in the per-ROI connection table.
    Connections outside of the primary ROIs are labeled with the special name
    `NotPrimary` (which is not currently an ROI name in neuprint itself).

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

        label:
            Either 'Neuron' or 'Segment' (which includes Neurons)

        client:
            If not provided, the global default :py:class:`.Client` will be used.

    Returns:
        Two DataFrames, ``(traced_neurons_df, roi_conn_df)``, containing a
        table of neuron IDs and the per-ROI connection table, respectively.
        See caveat above concerning non-primary ROIs.

    See also:
        :py:func:`.fetch_traced_adjacencies()`

    """
    assert (not isinstance(sources, type(None)) or not isinstance(targets, type(None))), \
              "Must provide either sources or targets or both."

    neurons_df = []
    if not isinstance(sources, type(None)):
        if not isinstance(sources, SegmentCriteria):
            # A previous version of this function accepted a list of bodyIds.
            # We still support that for now.
            assert isinstance(sources, collections.abc.Iterable), \
                f"Invalid criteria: {sources}"
            sources = SegmentCriteria(bodyId=sources)
        else:
            sources = copy.copy(sources)
            sources.matchvar = 'n'

        q = f"""\
            MATCH (n:{sources.label})
            {sources.all_conditions(prefix=8)}
            RETURN n.bodyId as bodyId, n.instance as instance, n.type as type
            ORDER BY n.bodyId
        """
        sources_df = client.fetch_custom(q)
        neurons_df.append(sources_df)

    if not isinstance(targets, type(None)):
        if not isinstance(targets, SegmentCriteria):
            # A previous version of this function accepted a list of bodyIds.
            # We still support that for now.
            assert isinstance(targets, collections.abc.Iterable), \
                f"Invalid criteria: {targets}"
            targets = SegmentCriteria(bodyId=targets)
        else:
            targets = copy.copy(targets)
            targets.matchvar = 'n'

        # Save time in cases where sources = targets
        if targets == sources:
            targets_df = sources_df
        else:
            q = f"""\
                MATCH (n:{targets.label})
                {targets.all_conditions(prefix=8)}
                RETURN n.bodyId as bodyId, n.instance as instance, n.type as type
                ORDER BY n.bodyId
            """
            targets_df = client.fetch_custom(q)
            neurons_df.append(targets_df)

    # Merge sources and targets
    neurons_df = pd.concat(neurons_df, ignore_index=True)
    neurons_df.drop_duplicates('bodyId', inplace=True)
    neurons_df.reset_index(drop=True, inplace=True)

    # If either sources or targets is not provided use the others label
    sources_label = sources.label if sources else targets.label
    targets_label = targets.label if targets else sources.label

    # Fetch connections in batches
    conn_tables = []
    sources_iter = sources_df.bodyId.values if sources else [None]
    targets_iter = targets_df.bodyId.values if targets else [None]

    total_chunks = math.ceil(len(sources_iter) / batch_size) * \
                   math.ceil(len(targets_iter) / batch_size)

    with tqdm(total=total_chunks,
              disable=total_chunks == 1,
              leave=False) as pbar:
        for sources_start in range(0, len(sources_iter), batch_size):
            sources_stop = sources_start + batch_size
            sources_batch = sources_iter[sources_start:sources_stop]
            for targets_start in range(0, len(targets_iter), batch_size):
                targets_stop = targets_start + batch_size
                targets_batch = targets_iter[targets_start:targets_stop]

                where = []
                if any(sources_batch):
                    where.append(f'n.bodyId in {sources_batch.tolist()}')
                if any(targets_batch):
                    where.append(f'm.bodyId in {targets_batch.tolist()}')

                q = f"""\
                    MATCH (n:{sources_label})-[e:ConnectsTo]->(m:{targets_label})
                WHERE {' AND '.join(where)}
                    RETURN n.bodyId as bodyId_pre, m.bodyId as bodyId_post, e.weight as weight, e.roiInfo as roiInfo
                """
                conn_tables.append(client.fetch_custom(q))
                pbar.update(1)
            pbar.update(1)

    # Combine batches
    connections_df = pd.concat(conn_tables, ignore_index=True)

    # Parse roiInfo json
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
    roi_conn_df = roi_conn_df.query('roi in @client.primary_rois')
    
    # Add a special roi name "NotPrimary" to account for the
    # difference between total weights and primary-only weights.
    primary_totals = roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()

    totals_df = connections_df.merge(primary_totals, 'left', on=['bodyId_pre', 'bodyId_post'], suffixes=['_all', '_primary'])
    totals_df.fillna(0, inplace=True)
    totals_df['weight_notprimary'] = totals_df.eval('weight_all - weight_primary')
    totals_df['roi'] = 'NotPrimary'
    
    # Drop weights other than NotPrimary
    totals_df = totals_df[['bodyId_pre', 'bodyId_post', 'roi', 'weight_notprimary']]
    totals_df = totals_df.rename(columns={'weight_notprimary': 'weight'})
    
    roi_conn_df = pd.concat((roi_conn_df, totals_df), ignore_index=True)
    roi_conn_df.sort_values(['bodyId_pre', 'bodyId_post', 'weight'], ascending=[True, True, False], inplace=True)
    roi_conn_df.reset_index(drop=True, inplace=True)
    
    # Double-check our math against the original totals
    summed_roi_weights = roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()
    compare_df = connections_df.merge(summed_roi_weights, 'left', on=['bodyId_pre', 'bodyId_post'], suffixes=['_orig', '_summed'])
    assert compare_df.fillna(0).eval('weight_orig == weight_summed').all()
    
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
    Convenience function for calling :py:func:`.fetch_adjacencies()`
    for traced, non-cropped neurons. 
 
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
    return fetch_adjacencies(criteria, criteria, export_dir, batch_size, client=client)


@inject_client
def fetch_meta(*, client=None):
    """
    Fetch the dataset metadata, parsing json fields where necessary.
    
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
    Fetch the list of all ROIs in the dataset,
    from the dataset metadata.
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
    Fetch the list of 'primary' ROIs in the dataset,
    from the dataset metadata.
    Primary ROIs do not overlap with each other.
    """
    q = "MATCH (m:Meta) RETURN m.primaryRois as rois"
    rois = client.fetch_custom(q)['rois'].iloc[0]
    return sorted(rois)


@inject_client
def fetch_synapses(segment_criteria, synapse_criteria=None, *, client=None):
    """
    Fetch synapses from neuron or selection of neurons.

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
        MATCH (n:{segment_criteria.label})-[:Contains]->(ss:SynapseSet),
              (ss)-[:Contains]->(s:Synapse)

        {segment_criteria.all_conditions(prefix=8)}
        {synapse_criteria.condition('n', 's', prefix=8)}
        
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
    Fetch a table of synapse-synapse connections between source and target neurons.
    
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
            ``upstream_roi`` and ``downstream_roi`` columns will contain a single
            string (or ``None``) in every row.
            
            Otherwise, the roi columns will contain a list of ROIs for every row.
            (Primary ROIs do not overlap, so every synapse resides in only one
            (or zero) primary ROI.)
            See :py:class:`.SynapseCriteria` for details.

        client:
            If not provided, the global default :py:class:`.Client` will be used.

    Returns:
    
        DataFrame in which each row represents a single synaptic connection
        between an upstream and downstream body.
        Synapse locations are listed in columns ``[ux, uy, uz]`` and ``[dx, dy, dz]``
        for the upstream and downstream syanpses, respectively.
        The ``upstream_roi`` and ``downstream_roi`` columns will contain either strings
        or lists-of-strings, depending on the ``primary_only`` synapse criteria as
        described above.
    
    Example:

        .. code-block:: ipython

            In [1]: from neuprint import fetch_synapse_connections, SegmentCriteria as SC, SynapseCriteria as SynC
               ...: fetch_synapse_connections(SC(bodyId=792368888), None, SynC(rois=['PED(R)', 'SMP(R)'], primary_only=True))
            Out[1]:
                upstream_bodyId  downstream_bodyId upstream_roi downstream_roi     ux     uy     uz     dx     dy     dz  upstream_confidence  downstream_confidence
            0         792368888          754547386       PED(R)         PED(R)  14013  27747  19307  13992  27720  19313                0.996               0.401035
            1         792368888          612742248       PED(R)         PED(R)  14049  27681  19417  14044  27662  19408                0.921               0.881487
            2         792368888         5901225361       PED(R)         PED(R)  14049  27681  19417  14055  27653  19420                0.921               0.436177
            3         792368888         5813117385       SMP(R)         SMP(R)  23630  29443  16297  23634  29437  16279                0.984               0.970746
            4         792368888         5813083733       SMP(R)         SMP(R)  23630  29443  16297  23634  29419  16288                0.984               0.933871
            5         792368888         5813058320       SMP(R)         SMP(R)  18662  34144  12692  18655  34155  12697                0.853               0.995000
            6         792368888         5812981989       PED(R)         PED(R)  14331  27921  20099  14351  27928  20085                0.904               0.877373
            7         792368888         5812981381       PED(R)         PED(R)  14331  27921  20099  14301  27919  20109                0.904               0.567321
            8         792368888         5812981381       PED(R)         PED(R)  14013  27747  19307  14020  27747  19285                0.996               0.697836
            9         792368888         5812979314       PED(R)         PED(R)  14331  27921  20099  14329  27942  20109                0.904               0.638362
            10        792368888          424767514       PED(R)         PED(R)  14331  27921  20099  14324  27934  20085                0.904               0.985734
            11        792368888          424767514       PED(R)         PED(R)  14013  27747  19307  14020  27760  19294                0.996               0.942831
            12        792368888          424767514       PED(R)         PED(R)  14049  27681  19417  14040  27663  19420                0.921               0.993586
            13        792368888          331662710       SMP(R)         SMP(R)  23630  29443  16297  23644  29429  16302                0.984               0.996389
            14        792368888         1196854070       PED(R)         PED(R)  14331  27921  20099  14317  27935  20101                0.904               0.968408
            15        792368888         1131831702       SMP(R)         SMP(R)  23630  29443  16297  23651  29434  16316                0.984               0.362952
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
        
        RETURN n.bodyId as upstream_bodyId,
               m.bodyId as downstream_bodyId,
               ns.location.x as ux,
               ns.location.y as uy,
               ns.location.z as uz,
               ms.location.x as dx,
               ms.location.y as dy,
               ms.location.z as dz,
               ns.confidence as up_conf,
               ms.confidence as dn_conf,
               apoc.map.removeKeys(ns, ['location', 'confidence', 'type']) as upstream_info,
               apoc.map.removeKeys(ms, ['location', 'confidence', 'type']) as downstream_info
    """)
    data = client.fetch_custom(cypher, format='json')['data']

    # Assemble DataFrame
    syn_table = []
    for upstream_body, downstream_body, ux, uy, uz, dx, dy, dz, up_conf, dn_conf, upstream_info, downstream_info in data:
        # Exclude non-primary ROIs if necessary
        up_rois = return_rois & {*upstream_info.keys()}
        dn_rois = return_rois & {*downstream_info.keys()}

        # Intern the ROIs to save RAM
        up_rois = sorted(map(sys.intern, up_rois))
        dn_rois = sorted(map(sys.intern, dn_rois))

        up_rois = up_rois or [None]
        dn_rois = dn_rois or [None]

        # Should be (at most) one ROI when primary_only=True,
        # so only show that one (not a list)
        if synapse_criteria.primary_only:
            up_rois = up_rois[0]
            dn_rois = dn_rois[0]
        
        syn_table.append((upstream_body, downstream_body, up_rois, dn_rois, ux, uy, uz, dx, dy, dz, up_conf, dn_conf))

    syn_df = pd.DataFrame(syn_table, columns=['upstream_bodyId', 'downstream_bodyId',
                                              'upstream_roi', 'downstream_roi',
                                              'ux', 'uy', 'uz', 'dx', 'dy', 'dz',
                                              'upstream_confidence', 'downstream_confidence'])

    # Save RAM with smaller dtypes
    syn_df['ux'] = syn_df['ux'].astype(np.int32)
    syn_df['uy'] = syn_df['uy'].astype(np.int32)
    syn_df['uz'] = syn_df['uz'].astype(np.int32)
    syn_df['dx'] = syn_df['dx'].astype(np.int32)
    syn_df['dy'] = syn_df['dy'].astype(np.int32)
    syn_df['dz'] = syn_df['dz'].astype(np.int32)
    syn_df['upstream_confidence'] = syn_df['upstream_confidence'].astype(np.float32)
    syn_df['downstream_confidence'] = syn_df['downstream_confidence'].astype(np.float32)

    return syn_df
