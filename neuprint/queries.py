import os

import numpy as np
import pandas as pd
from tqdm import trange

from .utils import make_iterable, make_args_iterable, where_expr
from .client import inject_client

try:
    # ujson is faster than Python's builtin json module;
    # use it if the user happens to have it installed.
    import ujson as json
except ImportError:
    import json


@inject_client
def fetch_custom(cypher, dataset="", format='pandas', *, client=None):
    """
    Alternative form of ``Client.fetch_custom()``, as a free function.
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
            If not provided, the global default ``Client`` will be used.
    
    Returns:
        Either json or DataFrame, depending on ``format``.
    """
    return client.fetch_custom(cypher, dataset, format)


@inject_client
@make_args_iterable(['bodyId', 'instance', 'type', 'status', 'inputRois', 'outputRois'])
def fetch_neurons(bodyId=None, instance=None, type=None, status=None, cropped=None,
                  inputRois=None, outputRois=None, min_pre=0, min_post=0,
                  regex=False, *, client=None):
    """
    Search for a set of neurons by bodyId, instance, roi, etc.
    Returns their properties, including the distibution of their synapses in all brain regions.
    
    This is the Python equivalent to the Neuprint Explorer `Find Neurons`_ page.

    Returns data in the the same format as :py:func:`find_custom_neurons()`,
    but doesn't require you to write cypher.
    
    .. _Find Neurons: https://neuprint.janelia.org/?dataset=hemibrain%3Av1.0&qt=findneurons&q=1
    
    Args:
        bodyId:
            Integer or list of ints.
        instance:
            str or list of strings
            If ``regex=True``, then the instance will be matched as a regular expression.
        type:
            str or list of strings
            If ``regex=True``, then the type will be matched as a regular expression.
        status:
            str or list of strings
        cropped:
            Boolean.
            If given, restrict results to neurons that are cropped or not.
        inputRoi:
            str or list of strings
            Only Neurons which have inputs in EVERY one of the given ROIs will be matched.
            ``regex`` does not apply to this parameter.
        outputRoi:
            str or list of strings
            Only Neurons which have outputs in EVERY one of the given ROIs will be matched.
            ``regex`` does not apply to this parameter.
        min_pre:
            int
            Exclude neurons that don't have at least this many t-bars (outputs).
        min_post:
            int
            Exclude neurons that don't have at least this many PSDs (inputs).
        regex:
            Boolean.
            If ``True``, the ``instance`` and ``type`` arguments will be interpreted as
            regular expressions, rather than exact match strings.

    Returns:
        Two DataFrames.
        ``(neurons_df, roi_counts_df)``
        
        In ``neurons_df``, all available columns ``:Neuron`` columns are returned, with the following changes:
        
            - ROI boolean columns are removed
            - ``roiInfo`` is parsed as json data
            - ``somaLocation`` is provided as a list ``[x, y, z]``
            - New columns ``input_rois`` and ``output_rois`` contain lists of each neuron's ROIs.
        
        In ``roi_counts_df``, the ``roiInfo`` has been loadded into a table
        of per-neuron-per-ROI synapse counts, with separate columns
        for ``pre`` (outputs) and ``post`` (inputs).

    See also:

        If you like the output format of this function but you want
        to provide your own cypher query, see :py:func:`fetch_custom_neurons()`.


    Example:
    
        .. code-block:: ipython
        
            In [1]: neurons_df, roi_counts_df = fetch_neurons(
               ...:     input_roi=['SIP(R)', 'aL(R)'], status='Traced',
               ...:     type='MBON.*', instance='.*', regex=True)
            
            In [2]: neurons_df.columns
            Out[2]:
            Index(['bodyId', 'status', 'cropped', 'type', 'instance', 'cellBodyFiber',
                   'somaRadius', 'somaLocation', 'size', 'pre', 'post', 'statusLabel',
                   'inputRois', 'outputRois', 'roiInfo'],
                  dtype='object')
            
            In [3]: neurons_df.iloc[:5, :11]
            Out[3]:
                  bodyId  status  cropped    type                     instance cellBodyFiber  somaRadius           somaLocation        size   pre   post
            0  300972942  Traced    False  MBON14                 MBON14(a3)_R           NaN         NaN                   None  1563154937   543  13634
            1  422725634  Traced    False  MBON06        MBON06(B1>a)(AVM07)_L           NaN         NaN                   None  3118269136  1356  20978
            2  423382015  Traced    False  MBON23        MBON23(a2sp)(PDL05)_R          SFS1       291.0   [7509, 13310, 14016]   857093893   733   4466
            3  423774471  Traced    False  MBON19       MBON19(a2p3p)(PDL05)_R          SFS1       286.0   [5459, 15006, 10552]   628019179   299   1484
            4  424767514  Traced    False  MBON11  MBON11(y1pedc>a/B)(ADM05)_R        mAOTU2       694.5  [18614, 35832, 19448]  5249327644  1643  27641
            
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
    inputRois = {*inputRois}
    outputRois = {*outputRois}

    assert len(bodyId) == 0 or np.issubdtype(np.asarray(bodyId).dtype, np.integer), \
        "bodyId should be an integer or list of integers"
    
    assert not regex or len(instance) <= 1, "Please provide only one regex pattern for instance"
    assert not regex or len(type) <= 1, "Please provide only one regex pattern for type"
    
    assert any([len(bodyId), len(instance), len(type), len(status),
                cropped is not None, len(inputRois), len(outputRois),
                min_pre, min_post]), \
        "Please provide at least one search argument!"
    
    # Verify ROI names against known ROIs.
    neuprint_rois = {*fetch_all_rois(client=client)}
    unknown_input_rois = inputRois - neuprint_rois
    if unknown_input_rois:
        raise RuntimeError(f"Unrecognized input ROIs: {unknown_input_rois}")

    unknown_output_rois = outputRois - neuprint_rois
    if unknown_output_rois:
        raise RuntimeError(f"Unrecognized output ROIs: {unknown_output_rois}")

    body_expr = where_expr('bodyId', bodyId)
    instance_expr = where_expr('instance', instance, regex)
    type_expr = where_expr('type', type, regex)
    status_expr = where_expr('status', status)
    
    if cropped is None:
        cropped_expr = ""
    elif cropped:
        cropped_expr = "n.cropped"
    else:
        # Not all neurons have the 'cropped' tag,
        # so simply checking for False values isn't enough.
        cropped_expr = "(NOT n.cropped OR NOT exists(n.cropped))"

    query_rois = {*inputRois, *outputRois}
    if query_rois:
        roi_expr = "(" + " AND ".join(f"n.`{roi}`" for roi in query_rois) + ")"
    else:
        roi_expr = ""

    if min_pre:
        pre_expr = f"n.pre >= {min_pre}"
    else:
        pre_expr = ""

    if min_post:
        post_expr = f"n.post >= {min_post}"
    else:
        post_expr = ""

    # Build WHERE clause by combining the exprs
    exprs = [body_expr, instance_expr, type_expr, status_expr,
             cropped_expr, roi_expr, pre_expr, post_expr]
    exprs = filter(None, exprs)

    WHERE =  "WHERE\n"
    WHERE += "          "
    WHERE += "\n          AND ".join(exprs)

    q = f"""\
        MATCH (n :Neuron)
        {WHERE}
        RETURN n
        ORDER BY n.bodyId
    """
    
    neuron_df, roi_counts_df = fetch_custom_neurons(q, neuprint_rois, client=client)
    if len(neuron_df) == 0:
        return neuron_df, roi_counts_df

    # Our query matched any neuron that intersected all of the ROIs,
    # without distinguishing between input and output.
    # Now filter the list to ensure that input/output requirements are respected.
    if inputRois:
        # Keep only neurons where every required input ROI is present as an input.
        num_missing = neuron_df['inputRois'].apply(lambda rowInputRois: len(inputRois - {*rowInputRois}))
        neuron_df = neuron_df.loc[(num_missing == 0)]

    if outputRois:
        # Keep only neurons where every required output ROI is present as an output.
        num_missing = neuron_df['outputRois'].apply(lambda rowOutputRois: len(outputRois - {*rowOutputRois}))
        neuron_df = neuron_df.loc[(num_missing == 0)]

    # Filter the ROI counts to exclude neurons that were removed above.
    _filtered_bodies = neuron_df['bodyId']
    roi_counts_df.query('bodyId in @_filtered_bodies', inplace=True)
    
    neuron_df = neuron_df.copy()
    return neuron_df, roi_counts_df



@inject_client
def fetch_custom_neurons(q, neuprint_rois=None, *, client=None):
    """
    Use a custom query to fetch a neuron table, with nicer output
    than you would get from a call to :py:func:`fetch_custom()`.
    
    Returns data in the the same format as :py:func:`fetch_neurons()`.
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
    neuron_df['roiInfo'] = neuron_df['roiInfo'].apply(lambda s: json.loads(s))
    
    if neuprint_rois is None:
        neuprint_rois = {*fetch_all_rois(client=client)}
    
    # Drop roi columns
    columns = {*neuron_df.columns} - neuprint_rois
    neuron_df = neuron_df[[*columns]]

    # Extract somaLocation
    if 'somaLocation' in neuron_df:
        no_soma = neuron_df['somaLocation'].isnull()
        neuron_df.loc[no_soma, 'somaLocation'] = None
        neuron_df.loc[~no_soma, 'somaLocation'] = neuron_df.loc[~no_soma, 'somaLocation'].apply(lambda sl: sl.get('coordinates'))
    
    # Make a list of rois for every neuron (both pre and post)
    neuron_df['inputRois'] = neuron_df['roiInfo'].apply(lambda d: sorted([k for k,v in d.items() if v.get('post')]))
    neuron_df['outputRois'] = neuron_df['roiInfo'].apply(lambda d: sorted([k for k,v in d.items() if v.get('pre')]))

    # Specify column order:
    # Standard columns first, than any extra columns in the results (if any).
    neuron_cols = [*filter(lambda c: c in neuron_df.columns, neuron_cols)]
    extra_cols = {*neuron_df.columns} - {*neuron_cols}
    neuron_cols += [*extra_cols]
    neuron_df = neuron_df[[*neuron_cols]]

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
@make_args_iterable(['upstream_bodyId', 'upstream_instance', 'upstream_type', 'downstream_bodyId', 'downstream_instance', 'downstream_type'])
def fetch_simple_connections(upstream_bodyId=None, upstream_instance=None, upstream_type=None,
                             downstream_bodyId=None, downstream_instance=None, downstream_type=None,
                             min_weight=1, label='Neuron',
                             regex=False, properties=['status', 'cropped', 'type', 'instance'],
                             *, client=None):
    """
    Find all connections from a set of "upstream" neurons,
    or to a set of "downstream" neurons,
    or all connections from a set of upstream neurons to a set of downstream neurons.

    Args:
        upstream_bodyId:
            Integer or list of ints.
        upstream_instance:
            str or list of strings
            If ``regex=True``, then the instance will be matched as a regular expression.
        upstream_type:
            str or list of strings
            If ``regex=True``, then the type will be matched as a regular expression.
        downstream_bodyId:
            Integer or list of ints.
        downstream_instance:
            str or list of strings
            If ``regex=True``, then the instance will be matched as a regular expression.
        downstream_type:
            str or list of strings
            If ``regex=True``, then the type will be matched as a regular expression.
        min_weight:
            Exclude connections below this weight.
        label:
            Return results for Neurons (default) or all Segments.
        properties:
            Additional columns to include in the results, for both the upstream and downstream body.
        regex:
            If ``True``, instance and type arguments will be interpreted as regular expressions.
    
    Returns:
        DataFrame
        One row per connection, with columns for upstream and downstream properties.
    """
    assert label in ('Neuron', 'Segment'), \
        f"Invalid node type: {label}"
    
    assert sum(map(len, [upstream_bodyId, upstream_instance, upstream_type,
                         downstream_bodyId, downstream_instance, downstream_type])) > 0, \
        "Need at least one input criteria"

    upstream_body_expr = where_expr('bodyId', upstream_bodyId, False, 'upstream')
    upstream_instance_expr = where_expr('instance', upstream_instance, regex, 'upstream')
    upstream_type_expr = where_expr('type', upstream_type, regex, 'upstream')

    downstream_body_expr = where_expr('bodyId', downstream_bodyId, False, 'downstream')
    downstream_instance_expr = where_expr('instance', downstream_instance, regex, 'downstream')
    downstream_type_expr = where_expr('type', downstream_type, regex, 'downstream')

    if min_weight > 1:
        weight_expr = f"e.weight >= {min_weight}"
    else:
        weight_expr = ""
    
    # Build WHERE clause by combining the exprs
    exprs = [upstream_body_expr, upstream_instance_expr, upstream_type_expr,
             downstream_body_expr, downstream_instance_expr, downstream_type_expr,
             weight_expr]
    exprs = filter(None, exprs)

    WHERE =  "WHERE "
    WHERE += "\n              AND ".join(exprs)

    return_props = ['upstream.bodyId', 'downstream.bodyId', 'e.weight as weight']
    if properties:
        return_props += [f'upstream.{p}' for p in properties]
        return_props += [f'downstream.{p}' for p in properties]
    
    return_props_str = ',\n               '.join(return_props)

    # If roiInfo is requested, convert from json
    return_props_str = return_props_str.replace('upstream.roiInfo',
                            'apoc.convert.fromJsonMap(upstream.roiInfo) as upstream_roiInfo')
    return_props_str = return_props_str.replace('downstream.roiInfo',
                            'apoc.convert.fromJsonMap(downstream.roiInfo) as downstream_roiInfo')

    q = f"""\
        MATCH (upstream:{label})-[e:ConnectsTo]->(downstream:{label})
        {WHERE}
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
def fetch_adjacencies(bodies, export_dir=None, batch_size=200, *, client=None):
    """
    Fetch the adjacency table for connections amongst a set of neurons, broken down by ROI.
    Synapses which do not fall on any ROI will be listed as having ROI 'None'.
    Only primary ROIs are included in the results.

    Args:
        bodies:
            Limit results to connections between the given bodyIds.
            If not provided, then use all non-cropped Traced neurons. 
            
        export_dir:
            Optional. Export CSV files for the neuron table,
            connection table (total weight), and connection table (per ROI).
            
        batch_size:
            For optimal performance, connections will be fetched in batches.
            This parameter specifies the batch size.
    
    Returns:
        Two DataFrames, ``(traced_neurons_df, roi_conn_df)``, containing the
        table of neuron IDs and the per-ROI connection table, respectively.
        Only primary ROIs are included in the per-ROI connection table.

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
    q = f"""\
        WITH {[*bodies]} as bodies
        MATCH (n:Neuron)
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
            MATCH (n:Neuron) - [e:ConnectsTo] -> (m:Neuron)
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
    # Fetch the list of primary ROIs
    q = """\
        MATCH (m:Meta)
        RETURN m.primaryRois as rois
    """
    primary_rois = client.fetch_custom(q)['rois'].iloc[0]
    roi_conn_df = roi_conn_df.query('roi in @primary_rois or roi == "None"')
    
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
        p = f"{export_dir}/traced-connections.csv"
        conn_groups = roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)
        total_conn_df = conn_groups['weight'].sum()
        total_conn_df.to_csv(p, index=False, header=True)

    return neurons_df, roi_conn_df


@inject_client
def fetch_traced_adjacencies(export_dir=None, batch_size=200, *, client=None):
    """
    Finds the set of all non-cropped traced neurons, and then
    calls :py:func:`fetch_adjacencies()`. 
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
def fetch_all_rois(*, client):
    """
    Fetch the list of all ROIs in the dataset,
    from the dataset metadata.
    """
    meta = fetch_meta(client=client)
    official_rois = {*meta['roiInfo'].keys()}

    # These two ROIs are special:
    # For historical reasons, they exist as tags, but are not listed in the Meta roiInfo.
    hidden_rois = {'FB-column3', 'AL-DC3'}

    return sorted(official_rois | hidden_rois)


@inject_client
def fetch_primary_rois(*, client):
    """
    Fetch the list of 'primary' ROIs in the dataset,
    from the dataset metadata.
    Primary ROIs do not overlap with each other.
    """
    q = "MATCH (m:Meta) RETURN m.primaryRois as rois"
    rois = client.fetch_custom(q)['rois'].iloc[0]
    return sorted(rois)

