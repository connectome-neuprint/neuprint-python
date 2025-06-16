import os
import copy
import warnings
from textwrap import indent, dedent

import ujson
import pandas as pd

from ..client import inject_client, NeuprintTimeoutError
from ..utils import ensure_list_args, trange
from .neuroncriteria import NeuronCriteria, neuroncriteria_args, copy_as_neuroncriteria


@inject_client
@ensure_list_args(['rois'])
@neuroncriteria_args('upstream_criteria', 'downstream_criteria')
def fetch_simple_connections(upstream_criteria=None, downstream_criteria=None, rois=None, min_weight=1,
                             properties=['type', 'instance'],
                             *, client=None):
    """
    Find connections to/from small set(s) of neurons.  Most users
    should prefer ``fetch_adjacencies()`` instead of this function.

    Finds all connections from a set of "upstream" neurons,
    or to a set of "downstream" neurons,
    or all connections from a set of upstream neurons to a set of downstream neurons.

    Note:
        This function is not intended to be used with very large sets of neurons.
        Furthermore, it does not return ROI information in a convenient format.
        But the simple output table it returns is sometimes convenient for small,
        interactive queries.

        To fetch all adjacencies between a large set of neurons,
        see :py:func:`fetch_adjacencies()`, which also has additional
        ROI-filtering options, and also returns ROI info in a separate table.

    Args:
        upstream_criteria (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            How to filter for neurons on the presynaptic side of connections.
        downstream_criteria (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            How to filter for neurons on the postsynaptic side of connections.
        rois:
            Limit results to neuron pairs that connect in at least one of the given ROIs.
            Note that the total weight of each connection may include connections outside of the listed ROIs, too.
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

            In [1]: from neuprint import fetch_simple_connections
               ...: sources = [329566174, 425790257, 424379864, 329599710]
               ...: targets = [425790257, 424379864, 329566174, 329599710, 420274150]
               ...: fetch_simple_connections(sources, targets)
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
    NC = NeuronCriteria
    up_crit = copy.deepcopy(upstream_criteria)
    down_crit = copy.deepcopy(downstream_criteria)

    if up_crit is None:
        up_crit = NC(label='Neuron', client=client)
    if down_crit is None:
        down_crit = NC(label='Neuron', client=client)

    up_crit.matchvar = 'n'
    down_crit.matchvar = 'm'

    assert up_crit is not None or down_crit is not None, "No criteria specified"

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

    combined_global_with = NC.combined_global_with([up_crit, down_crit], prefix=8)
    combined_conditions = NC.combined_conditions([up_crit, down_crit], ('n', 'm', 'e'), prefix=8)
    q = f"""\
        {combined_global_with}
        MATCH (n:{up_crit.label})-[e:ConnectsTo]->(m:{down_crit.label})

        {combined_conditions}

        WITH n, m, e
        WHERE e.weight >= {min_weight}

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
@ensure_list_args(['rois'])
@neuroncriteria_args('sources', 'targets')
def fetch_adjacencies(sources=None, targets=None, rois=None, min_roi_weight=1, min_total_weight=1,
                      include_nonprimary=False, export_dir=None, batch_size=200,
                      properties=['type', 'instance'], *, client=None):
    """
    Find connections to/from large sets of neurons, with per-ROI connection strengths.

    Fetches the adjacency table for connections between sets of neurons, broken down by ROI.
    Unless ``include_nonprimary=True``, only primary ROIs are included in the per-ROI connection table.
    Connections outside of the primary ROIs are labeled with the special name
    ``"NotPrimary"`` (which is not currently an ROI name in neuprint itself).

    Note:
        :py:func:`.fetch_simple_connections()` has similar functionality,
        but that function isn't suitable for querying large sets of neurons.
        However, it may be more convenient for small interactive queries.

    Args:
        sources (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            Limit results to connections from bodies that match this criteria.
            If ``None``, all neurons upstream of ``targets`` will be fetched.

        targets (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            Limit results to connections to bodies that match this criteria.
            If ``None``, all neurons downstream of ``sources`` will be fetched.

        rois:
            Limit results to connections within the listed ROIs.

        min_roi_weight:
            Limit results to connections of at least this strength within at least one of the returned ROIs.

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

        export_dir:
            Optional. Export CSV files for the neuron table,
            connection table (total weight), and connection table (per ROI).

        batch_size:
            For optimal performance, connections will be fetched in batches.
            This parameter specifies the batch size.

        properties:
            Which Neuron properties to include in the output table.

        client:
            If not provided, the global default :py:class:`.Client` will be used.

    Returns:
        Two DataFrames, ``(neurons_df, roi_conn_df)``, containing a
        table of neuron IDs and the per-ROI connection table, respectively.
        See caveat above concerning non-primary ROIs.

    See also:
        :py:func:`.fetch_simple_connections()`
        :py:func:`.fetch_traced_adjacencies()`

    Example:

        .. code-block:: ipython

            In [1]: from neuprint import Client
               ...: c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1')

            In [2]: from neuprint import fetch_adjacencies
               ...: sources = [329566174, 425790257, 424379864, 329599710]
               ...: targets = [425790257, 424379864, 329566174, 329599710, 420274150]
               ...: neuron_df, connection_df = fetch_adjacencies(sources, targets)

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
    ## Why is this function so dang long and complicated?
    ## --------------------------------------------------
    ##
    ## 1. It batches the requests.  Instead of fetching all adjacencies between
    ##    sources and targets at once, it splits the requests up into batches of
    ##    source (or target) bodies.
    ##
    ## 2. To achieve (1), it has to pre-fetch either the source body list or the
    ##    target body list.
    ##
    ## 3. To achieve (2), it first fetches the *counts* of the source/body lists,
    ##    and determines which is shorter, or at least which one can be fetched
    ##    within a short timeout.
    ##
    ## 4. It 'reshapes' roi info into a column, with special care given to the
    ##    `include_nonprimary` option, and also invents a special ROI `NotPrimary`.
    ##
    ## 5. It updates the neuron list if necessary to include all sources and targets.
    ##
    ## 6. It writes to CSV.

    ##
    ## Preprocess arguments
    ##

    rois = {*rois}
    invalid_rois = rois - {*client.all_rois}
    assert not invalid_rois, f"Unrecognized ROIs: {invalid_rois}"

    nonprimary_rois = rois - {*client.primary_rois}
    assert include_nonprimary or not nonprimary_rois, \
        f"Since you listed nonprimary rois ({nonprimary_rois}), please specify include_nonprimary=True"

    min_roi_weight = max(min_roi_weight, 1)
    min_total_weight = max(min_total_weight, min_roi_weight)

    if 'bodyId' not in properties:
        properties = ['bodyId'] + properties

    def _prepare_criteria(criteria, matchvar):
        criteria.matchvar = matchvar

        # If the user wants to filter for specific rois,
        # we can speed up the query by adding them to the NeuronCriteria
        if rois and not criteria.rois:
            criteria.rois = rois
            criteria.roi_req = 'any'

        return criteria

    # Ensure sources/targets are NeuronCriteria
    sources = _prepare_criteria(sources, 'n')
    targets = _prepare_criteria(targets, 'm')

    def _fetch_neurons(criteria):
        matchvar = criteria.matchvar

        return_props = [f'{matchvar}.{prop} as {prop}' for prop in properties]
        return_props = indent(',\n'.join(return_props), ' '*19)[19:]

        q = f"""\
            {criteria.global_with(prefix=12)}
            MATCH ({matchvar}:{criteria.label})
            {criteria.all_conditions(prefix=12)}
            WITH {matchvar}
            RETURN {return_props}
            ORDER BY bodyId
        """
        return client.fetch_custom(q)

    ##
    ## Pre-fetch either source list or target list (whichever is shorter)
    ##

    def _prefetch_batchlist():
        """
        Figure out whether 'sources' or 'targets' shorter,
        and fetch those bodies and return them.
        Return 'None' for the other list.
        """
        def _fetch_count(criteria, timeout):
            matchvar = criteria.matchvar
            q = f"""\
                CALL apoc.cypher.runTimeboxed("
                    {criteria.global_with(prefix=20)}
                    MATCH ({matchvar}:{criteria.label})
                    {criteria.all_conditions(prefix=20)}
                    RETURN count({matchvar}) as c
                ", {{}}, {timeout*1000}) YIELD value
                RETURN value.c as count
            """
            try:
                result = client.fetch_custom(q)['count']
            except NeuprintTimeoutError:
                return None

            if len(result) == 0:
                return None

            return result.iloc[0]

        num_sources = _fetch_count(sources, 5)
        num_targets = _fetch_count(targets, 5)

        if num_sources is None and num_targets is None:
            num_sources = _fetch_count(sources, 120)
            num_targets = _fetch_count(targets, 120)

        if num_sources is None and num_targets is None:
            raise RuntimeError("Both source and target list are too large to pre-fetch without timing out. "
                               "This query is too big to process.")

        if num_sources == 0:
            raise RuntimeError("No neurons match your source criteria")

        if num_targets == 0:
            raise RuntimeError("No neurons match your target criteria")

        sources_df = targets_df = None
        if (num_sources is not None) and (num_targets is not None):
            if num_sources <= num_targets:
                sources_df = _fetch_neurons(sources)
            else:
                targets_df = _fetch_neurons(targets)
        elif num_sources is not None:
            sources_df = _fetch_neurons(sources)
        elif num_targets is not None:
            targets_df = _fetch_neurons(targets)

        assert (sources_df is None) != (targets_df is None)
        return sources_df, targets_df

    sources_df, targets_df = _prefetch_batchlist()

    ##
    ## Fetch connections in batches
    ##

    def _fetch_connections():
        if rois:
            min_edge_weight = min_total_weight
        else:
            # If rois aren't specified, then we'll include 'NotPrimary' counts,
            # and that means we can't filter by weight in the query.
            # We'll filter afterwards, but here we can at least filter out 0-weight edges.
            min_edge_weight = 1

        # Fetch connections by batching either the source list
        # or the target list, not both.
        # (It turns out that batching across BOTH sources and
        # targets is much slower than batching across only one.)
        conn_tables = []

        if sources_df is not None:
            # Break sources into batches
            for batch_start in trange(0, len(sources_df), batch_size, disable=not client.progress):
                batch_stop = batch_start + batch_size
                source_bodies = sources_df['bodyId'].iloc[batch_start:batch_stop].tolist()

                batch_criteria = copy.copy(sources)
                batch_criteria.bodyId = source_bodies

                criteria_globals = [*batch_criteria.global_vars().keys(), *targets.global_vars().keys()]

                q = f"""\
                    {NeuronCriteria.combined_global_with((batch_criteria, targets), prefix=20)}
                    MATCH (n:{sources.label})-[e:ConnectsTo]->(m:{targets.label})
                    {batch_criteria.all_conditions(*'nme', *criteria_globals, prefix=20)}

                    // Artificial break in the query flow to fool the query
                    // planner into avoiding a Cartesian product.
                    // This improves performance considerably in some cases.
                    WITH {','.join([*'nme', *criteria_globals])}, true as _

                    {targets.all_conditions(*'nme', prefix=20)}

                    // -- Filter by total connection weight --
                    WITH n,m,e
                    WHERE e.weight >= {min_edge_weight}

                    RETURN n.bodyId as bodyId_pre,
                           m.bodyId as bodyId_post,
                           e.weight as weight,
                           e.roiInfo as roiInfo
                """
                t = client.fetch_custom(q)
                if len(t) > 0:
                    conn_tables.append(t)
        else:
            # Break targets into batches
            for batch_start in trange(0, len(targets_df), batch_size, disable=not client.progress):
                batch_stop = batch_start + batch_size
                target_bodies = targets_df['bodyId'].iloc[batch_start:batch_stop].tolist()

                batch_criteria = copy.copy(targets)
                batch_criteria.bodyId = target_bodies

                criteria_globals = [*batch_criteria.global_vars().keys(), *sources.global_vars().keys()]

                q = f"""\
                    {NeuronCriteria.combined_global_with((sources, batch_criteria), prefix=20)}
                    MATCH (n:{sources.label})-[e:ConnectsTo]->(m:{targets.label})
                    {batch_criteria.all_conditions(*'nme', *criteria_globals, prefix=20)}

                    // Artificial break in the query flow to fool the query
                    // planner into avoiding a Cartesian product.
                    // This improves performance considerably in some cases.
                    WITH {','.join([*'nme', *criteria_globals])}, true as _

                    {sources.all_conditions(*'nme', prefix=20)}

                    // -- Filter by total connection weight --
                    WITH n,m,e
                    WHERE e.weight >= {min_edge_weight}

                    RETURN n.bodyId as bodyId_pre,
                           m.bodyId as bodyId_post,
                           e.weight as weight,
                           e.roiInfo as roiInfo
                """
                t = client.fetch_custom(q)
                if len(t) > 0:
                    conn_tables.append(t)

        if not conn_tables:
            return []

        # Combine batches
        connections_df = pd.concat(conn_tables, ignore_index=True)
        return connections_df

    connections_df = _fetch_connections()
    if len(connections_df) == 0:
        # Return empty DataFrames, but with the correct dtypes
        neuron_df = pd.DataFrame([], columns=['bodyId', 'instance', 'type'])
        neuron_df = neuron_df.astype({'bodyId': int, 'instance': str, 'type': str})
        roi_conn_df = pd.DataFrame([], columns=['bodyId_pre', 'bodyId_post', 'roi', 'weight'])
        roi_conn_df = roi_conn_df.astype({'bodyId_pre': int, 'bodyId_post': int, 'roi': str, 'weight': int})
        return neuron_df, roi_conn_df

    ##
    ## Post-process connections, construct roi_conn_df
    ##

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
    primary_totals = primary_roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'])['weight'].sum().reset_index()

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
                            .groupby(['bodyId_pre', 'bodyId_post'])['weight']
                            .sum()
                            .reset_index())
    compare_df = connections_df.merge(summed_roi_weights, 'left', on=['bodyId_pre', 'bodyId_post'], suffixes=['_orig', '_summed'])
    compare_df = compare_df.fillna(0)[['weight_orig', 'weight_summed']]
    mismatches = compare_df.eval('weight_orig != weight_summed')
    if mismatches.any():
        warnings.warn(
            "There appears to be an inconsistency in the neuprint data.\n"
            "Detected edge(s) in which the aggregate 'weight' does not match the sum of the roiInfo 'post' counts.\n"
            "Please report this to the neuprint administrators.\n"
            f"{compare_df.loc[mismatches]}"
        )

    # Filter for the user's ROIs, if any
    if rois:
        roi_conn_df.query('roi in @rois and weight > 0', inplace=True)

    if min_total_weight >= 1:
        total_weights_df = roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'])['weight'].sum().reset_index()
        keep_conns = total_weights_df.query('weight >= @min_total_weight')[['bodyId_pre', 'bodyId_post']]
        roi_conn_df = roi_conn_df.merge(keep_conns, 'inner', on=['bodyId_pre', 'bodyId_post'])

    # This is necessary, even if min_roi_weight == 1, to filter out zeros
    # that can occur in the case of weak inter-ROI connnections.
    roi_conn_df.query('weight >= @min_roi_weight', inplace=True)
    roi_conn_df.reset_index(drop=True, inplace=True)

    ##
    ## Construct neurons_df
    ##

    connected_bodies = pd.unique(roi_conn_df[['bodyId_pre', 'bodyId_post']].values.reshape(-1))

    # We only fetched either the source list or the target list.
    # we need to fetch the missing info based on the adjacencies we
    # actually found, and fetch it in batches.
    if sources_df is None:
        neurons_df = targets_df.query('bodyId in @connected_bodies')
        missing_label = sources.label
    else:
        neurons_df = sources_df.query('bodyId in @connected_bodies')
        missing_label = targets.label

    missing_bodies = [*set(connected_bodies) - set(neurons_df['bodyId'])]

    batches = []
    for start in trange(0, len(missing_bodies), 10_000, disable=not client.progress):
        batch_bodies = missing_bodies[start:start+10_000]
        batch_df = _fetch_neurons(NeuronCriteria(bodyId=batch_bodies, label=missing_label, client=client))
        batches.append( batch_df )

    neurons_df = pd.concat((neurons_df, *batches), ignore_index=True)
    neurons_df.reset_index(drop=True, inplace=True)

    ##
    ## Export to CSV
    ##
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

            In [1]: from neuprint import fetch_traced_adjacencies

            In [2]: neurons_df, roi_conn_df = fetch_traced_adjacencies('exported-connections')

            In [3]: roi_conn_df.head()
            Out[3]:
                   bodyId_pre  bodyId_post        roi  weight
            0      5813009352    516098538     SNP(R)       2
            1      5813009352    516098538     SLP(R)       2
            2       326119769    516098538     SNP(R)       1
            3       326119769    516098538     SLP(R)       1
            4       915960391    202916528         FB       1

            In [4]: # Obtain total weights (instead of per-connection-per-ROI weights)
               ...: conn_groups = roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)
               ...: total_conn_df = conn_groups['weight'].sum()
               ...: total_conn_df.head()
            Out[4]:
               bodyId_pre  bodyId_post  weight
            0   202916528    203253253       2
            1   202916528    203257652       2
            2   202916528    203598557       2
            3   202916528    234292899       4
            4   202916528    264986706       2
     """
    criteria = NeuronCriteria(status="Traced", cropped=False, client=client)
    return fetch_adjacencies(criteria, criteria, include_nonprimary=False, export_dir=export_dir, batch_size=batch_size, client=client)


@inject_client
@neuroncriteria_args('criteria')
def fetch_common_connectivity(criteria, search_direction='upstream', min_weight=1, properties=['type', 'instance'], *, client=None):
    """
    Find shared connections among a set of neurons.

    Given a set of neurons that match the given criteria, find neurons
    that connect to ALL of the neurons in the set, i.e. connections
    that are common to all neurons in the matched set.

    This is the Python equivalent to the Neuprint Explorer `Common Connectivity`_ page.

    .. _Common Connectivity: https://neuprint.janelia.org/?dataset=hemibrain%3Av1.2.1&qt=commonconnectivity&q=1


    Args:
        criteria (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            Used to determine the match set, for which common connections will be found.

        search_direction (``"upstream"`` or ``"downstream"``):
            Whether or not to search for common connections upstream of
            the matched neurons or downstream of the matched neurons.

        min_weight:
            Connections below the given strength will not be included in the results.

        properties:
            Additional columns to include in the results, for both the upstream and downstream body.

        client:
            If not provided, the global default :py:class:`.Client` will be used.

    Returns:
        DataFrame.
        (Same format as returned by :py:func:`fetch_simple_connections()`.)
        One row per connection, with columns for upstream and downstream properties.
        For instance, if ``search_direction="upstream"``, then the matched neurons will appear
        in the ``_post`` columns, and the common connections will appear in the ``_pre``
        columns.
    """
    assert search_direction in ('upstream', 'downstream')
    if search_direction == "upstream":
        edges_df = fetch_simple_connections(None, criteria, min_weight=min_weight, properties=properties, client=client)

        # How bodies many met main search criteria?
        num_primary = edges_df['bodyId_post'].nunique()

        # upstream bodies that connect to ALL of the main bodies are the 'common' bodies.
        upstream_counts = edges_df['bodyId_pre'].value_counts()
        _keep = upstream_counts[upstream_counts == num_primary].index
        return edges_df.query('bodyId_pre in @_keep')

    if search_direction == "downstream":
        edges_df = fetch_simple_connections(criteria, None, min_weight=min_weight, properties=properties, client=client)

        # How bodies many met main search criteria?
        num_primary = edges_df['bodyId_pre'].nunique()

        # upstream bodies that connect to ALL of the main are the 'common' bodies.
        upstream_counts = edges_df['bodyId_post'].value_counts()
        _keep = upstream_counts[upstream_counts == num_primary].index  # noqa
        return edges_df.query('bodyId_post in @_keep')


def fetch_shortest_paths(upstream_bodyId, downstream_bodyId, min_weight=1,
                        intermediate_criteria=None,
                        timeout=5.0, *,
                        client=None):
    """
    Find all neurons along the shortest path between two neurons.

    This function is a convenience wrapper around :py:func:`fetch_paths()`
    that sets ``path_length=None`` so the shortest path is returned.
    """

    return fetch_paths(upstream_bodyId, downstream_bodyId, min_weight=min_weight,
                        intermediate_criteria=intermediate_criteria,
                        timeout=timeout,
                        path_length=None, max_path_length=None,
                        client=client)


@inject_client
def fetch_paths(upstream_bodyId, downstream_bodyId, min_weight=1,
                intermediate_criteria=None,
                timeout=5.0, *,
                path_length=None, max_path_length=None,
                client=None):
    """
    Find all neurons along one or more paths between two neurons, as specified.

    Args:
        upstream_bodyId:
            The starting neuron

        downstream_bodyId:
            The destination neuron

        min_weight:
            Minimum connection strength for each step in the path.

        intermediate_criteria (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            Filtering criteria for neurons on path.
            All intermediate neurons in the path must satisfy this criteria.
            By default, ``NeuronCriteria(status="Traced")`` is used.

        path_length:
            The exact length of the path (number of relationships). If None, the shorted
            path will be returned. Only one of the two arguments
            ``path_length`` and ``max_path_length`` should be specified.

        max_path_length:
            The max length of the path (number of relationships). Only one of the two arguments
            ``path_length`` and ``max_path_length`` should be specified.

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

            In [1]: from neuprint import fetch_shortest_paths
               ...: fetch_shortest_paths(329566174, 294792184, min_weight=10)
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

    if path_length is not None and max_path_length is not None:
        raise ValueError("Please specify either 'path_length' or 'max_path_length', but not both.")

    if max_path_length is not None:
        path_mode = 'max'
    elif path_length is not None:
        path_mode = 'exact'
        try:
            path_length = int(path_length)
        except ValueError:
            path_length = None
            raise ValueError(f"Invalid path_length: {path_length}. "
                             "Please specify an integer value for path_length.")
    else:
        # default to shortest path
        path_mode = 'shortest'

    if intermediate_criteria is None:
        intermediate_criteria = NeuronCriteria(status="Traced", client=client)
    else:
        intermediate_criteria = copy_as_neuroncriteria(intermediate_criteria)

    assert len(intermediate_criteria.inputRois) == 0 and len(intermediate_criteria.outputRois) == 0, \
        "This function doesn't support search criteria that specifies inputRois or outputRois. "\
        "You can specify generic (intersecting) rois, though."

    intermediate_criteria.matchvar = 'n'

    timeout_ms = int(1000*timeout)

    nodes_where = intermediate_criteria.all_conditions(comments=False)
    if nodes_where:
        nodes_where += f"\n OR n.bodyId in [{upstream_bodyId}, {downstream_bodyId}]"
        nodes_where = nodes_where.replace('\n', '')
    else:
        # Even if there are no constraints whatsoever, we still need
        # an expression to serve as the predicate in the query below.
        nodes_where = "WHERE TRUE"

    if path_mode == 'shortest':
        path_clause = "allShortestPaths((src)-[:ConnectsTo*]->(dest))"
        return_clause1 = ""
        return_clause2 = ""
    else:
        if path_mode == 'exact':
            path_clause = f"(src)-[:ConnectsTo*{path_length}]->(dest)"
        elif path_mode == 'max':
            path_clause = f"(src)-[:ConnectsTo*1..{max_path_length}]->(dest)"
        return_clause1 = ",\n                   length(p) AS path_length"
        return_clause2 = ", value.path_length as path_length"

    q = f"""\
        call apoc.cypher.runTimeboxed(
            "{intermediate_criteria.global_with(prefix=12)}
            MATCH (src :Neuron {{ bodyId: {upstream_bodyId} }}),
                   (dest:Neuron {{ bodyId: {downstream_bodyId} }}),
                   p = {path_clause}

            WHERE     ALL (x in relationships(p) WHERE x.weight >= {min_weight})
                  AND ALL (n in nodes(p) {nodes_where})

            RETURN [n in nodes(p) | [n.bodyId, n.type]] AS path,
                   [x in relationships(p) | x.weight] AS weights{return_clause1}",

            {{}},{timeout_ms}) YIELD value
            RETURN value.path as path, value.weights AS weights{return_clause2}
    """
    results_df = client.fetch_custom(q)

    table_indexes = []
    table_bodies = []
    table_types = []
    table_weights = []
    table_path_lengths = []

    for path_index, (path, weights, *p_length) in enumerate(results_df.itertuples(index=False)):
        bodies, types = zip(*path)
        weights = [0, *weights]

        table_indexes += len(bodies) * [path_index]
        table_bodies += bodies
        table_types += types
        table_weights += weights
        if path_mode != 'shortest':
            table_path_lengths += len(bodies) * [p_length[0]]

    if path_mode == 'shortest':
        paths_df = pd.DataFrame({'path': table_indexes,
                                 'bodyId': table_bodies,
                                 'type': table_types,
                                 'weight': table_weights})
    else:
        paths_df = pd.DataFrame({'path': table_indexes,
                                 'bodyId': table_bodies,
                                 'type': table_types,
                                 'weight': table_weights,
                                 'path_length': table_path_lengths
                                 })

    return paths_df
