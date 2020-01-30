import os
import copy
from textwrap import indent, dedent

import pandas as pd
from tqdm import trange

from .client import inject_client
from .segmentcriteria import SegmentCriteria

try:
    # ujson is faster than Python's builtin json module;
    # use it if the user happens to have it installed.
    import ujson as json
except ImportError:
    import json


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
    neuron_df['roiInfo'] = neuron_df['roiInfo'].apply(lambda s: json.loads(s))
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
def fetch_adjacencies(bodies, export_dir=None, batch_size=200, label='Neuron', *, client=None):
    """
    Fetch the adjacency table for connections amongst a set of neurons, broken down by ROI.
    Only primary ROIs are included in the results.
    Synapses which do not fall on any primary ROI are not listed in the per-ROI table.

    Args:
        bodies:
            Limit results to connections between the given bodyIds.
            
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
        Two DataFrames, ``(traced_neurons_df, roi_conn_df)``, containing the
        table of neuron IDs and the per-ROI connection table, respectively.
        Only primary ROIs are included in the per-ROI connection table.

    See also:
        :py:func:`.fetch_traced_adjacecies()`
    """
    assert label in ('Neuron', 'Segment'), f"Invalid label: {label}"
    q = f"""\
        WITH {[*bodies]} as bodies
        MATCH (n:{label})
        WHERE n.bodyId in bodies
        RETURN n.bodyId as bodyId, n.instance as instance, n.type as type
    """
    neurons_df = client.fetch_custom(q)
    
    # Fetch connections in batches
    conn_tables = []
    for start in trange(0, len(neurons_df), batch_size):
        stop = start + batch_size
        batch_neurons = neurons_df['bodyId'].iloc[start:stop].tolist()
        q = f"""\
            MATCH (n:{label})-[e:ConnectsTo]->(m:{label})
            WHERE n.bodyId in {batch_neurons} AND m.status = "Traced" AND (not m.cropped)
            RETURN n.bodyId as bodyId_pre, m.bodyId as bodyId_post, e.weight as weight, e.roiInfo as roiInfo
        """
        conn_tables.append( client.fetch_custom(q) )
    
    # Combine batches
    connections_df = pd.concat(conn_tables, ignore_index=True)
    
    # Parse roiInfo json
    connections_df['roiInfo'] = connections_df['roiInfo'].apply(json.loads)

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
    
    # Export to CSV
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

        # Export Nodes
        p = f"{export_dir}/traced-neurons.csv"
        neurons_df.to_csv(p, index=False, header=True)
        
        # Export Edges (per ROI)
        p = f"{export_dir}/traced-roi-connections.csv"
        roi_conn_df.to_csv(p, index=False, header=True)

        # Export Edges (total weight)
        p = f"{export_dir}/traced-total-connections.csv"
        connections_df[['bodyId_pre', 'bodyId_post', 'weight']].to_csv(p, index=False, header=True)

    return neurons_df, roi_conn_df


@inject_client
def fetch_traced_adjacencies(export_dir=None, batch_size=200, *, client=None):
    """
    Finds the set of all non-cropped traced neurons, and then
    calls :py:func:`.fetch_adjacencies()`. 
 
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
    # Fetch the list of traced, non-cropped Neurons
    q = """\
        MATCH (n:Neuron)
        WHERE n.status = "Traced" AND (not n.cropped)
        RETURN n.bodyId as bodyId
    """
    bodies = client.fetch_custom(q)['bodyId']
    return fetch_adjacencies(bodies, export_dir, batch_size, client=client)


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

