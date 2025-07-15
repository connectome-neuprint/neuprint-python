from collections.abc import Container

import pandas as pd

from ..client import inject_client
from ..utils import trange
from .neuroncriteria import NeuronCriteria, neuroncriteria_args
from .general import fetch_custom
from .connectivity import fetch_adjacencies


@inject_client
@neuroncriteria_args('criteria')
def fetch_output_completeness(criteria, complete_statuses=['Traced'], batch_size=1000, *, client=None):
    """
    Compute an estimate of "output completeness" for a set of neurons.
    Output completeness is defined as the fraction of post-synaptic
    connections which belong to 'complete' neurons, as defined by their status.

    Args:
        criteria (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            Defines the set of neurons for which output completeness should be computed.

        complete_statuses:
            A list of neuron statuses should be considered complete for the purposes of this function.

    Returns:
        DataFrame with columns ``['bodyId', 'completeness', 'traced_weight', 'untraced_weight', 'total_weight']``

        For the purposes of these results, any statuses in the
        set of complete_statuses are considered 'traced'.
    """
    assert isinstance(criteria, NeuronCriteria)
    criteria.matchvar = 'n'

    assert isinstance(complete_statuses, Container)
    complete_statuses = list(complete_statuses)

    if batch_size is None:
        return _fetch_output_completeness(criteria, client)

    q = f"""\
        {criteria.global_with(prefix=8)}
        MATCH (n:{criteria.label})
        {criteria.all_conditions(prefix=8)}
        RETURN n.bodyId as bodyId
    """
    bodies = fetch_custom(q)['bodyId']

    batch_results = []
    for start in trange(0, len(bodies), batch_size, disable=not client.progress):
        criteria.bodyId = bodies[start:start+batch_size]
        _df = _fetch_output_completeness(criteria, complete_statuses, client)
        if len(_df) > 0:
            batch_results.append( _df )
    return pd.concat( batch_results, ignore_index=True )


def _fetch_output_completeness(criteria, complete_statuses, client=None):
    q = f"""\
        {criteria.global_with(prefix=8)}
        MATCH (n:{criteria.label})
        {criteria.all_conditions(prefix=8)}

        // Total connection weights
        MATCH (n)-[e:ConnectsTo]->(:Segment)
        WITH n, sum(e.weight) as total_weight

        // Traced connection weights
        MATCH (n)-[e2:ConnectsTo]->(m:Segment)
        WHERE
            m.status in {complete_statuses}
            OR m.statusLabel in {complete_statuses}

        RETURN n.bodyId as bodyId,
               total_weight,
               sum(e2.weight) as traced_weight
    """
    completion_stats_df = client.fetch_custom(q)
    completion_stats_df['untraced_weight'] = completion_stats_df.eval('total_weight - traced_weight')
    completion_stats_df['completeness'] = completion_stats_df.eval('traced_weight / total_weight')

    return completion_stats_df[['bodyId', 'total_weight', 'traced_weight', 'untraced_weight', 'completeness']]


@inject_client
@neuroncriteria_args('criteria')
def fetch_downstream_orphan_tasks(criteria, complete_statuses=['Traced'], *, client=None):
    """
    Fetch the set of "downstream orphans" for a given set of neurons.

    Returns a single DataFrame, where the downstream orphans have
    been sorted by the weight of the connection, and their cumulative
    contributions to the overall output-completeness of the upstream
    neuron is also given.

    That is, if you started tracing orphans from this DataFrame in
    order, then the ``cum_completeness`` column indicates how complete
    the upstream body is after each orphan becomes traced.

    Args:
        criteria (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            Determines the set of "upstream" bodies for which
            downstream orphans should be identified.

    Returns:
        DataFrame, where ``bodyId_pre`` contains the upstream bodies you specified
        via ``criteria``, and ``bodyId_post`` contains the list of downstream orphans.

    Example:

        .. code-block:: ipython

            In [1]: orphan_tasks = fetch_downstream_orphan_tasks(NC(status='Traced', cropped=False, rois=['PB']))

            In [1]: orphan_tasks.query('cum_completeness < 0.2').head(10)
            Out[1]:
                   bodyId_pre  bodyId_post  orphan_weight status_post  total_weight  orig_traced_weight  orig_untraced_weight  orig_completeness  cum_orphan_weight  cum_completeness
            6478    759685279    759676733              2      Assign          7757                1427                  6330           0.183963                  2          0.184221
            8932    759685279    913193340              1        None          7757                1427                  6330           0.183963                  3          0.184350
            8943    759685279    913529796              1        None          7757                1427                  6330           0.183963                  4          0.184479
            8950    759685279    913534416              1        None          7757                1427                  6330           0.183963                  5          0.184607
            12121  1002507170   1387701052              1        None           522                 102                   420           0.195402                  1          0.197318
            35764   759685279    790382544              1        None          7757                1427                  6330           0.183963                  6          0.184736
            36052   759685279    851023555              1      Assign          7757                1427                  6330           0.183963                  7          0.184865
            36355   759685279    974908767              2        None          7757                1427                  6330           0.183963                  9          0.185123
            36673   759685279   1252526211              1        None          7757                1427                  6330           0.183963                 10          0.185252
            44840   759685279   1129418900              1        None          7757                1427                  6330           0.183963                 11          0.185381

    """
    # Find all downstream segments, along with the status of all upstream and downstream bodies.
    status_df, roi_conn_df = fetch_adjacencies(criteria, NeuronCriteria(label='Segment', client=client), properties=['status', 'statusLabel'], client=client)

    # That table is laid out per-ROI, but we don't care about ROI. Aggregate.
    conn_df = roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'])['weight'].sum().reset_index()

    # Sort connections from strong to weak.
    conn_df.sort_values(['bodyId_pre', 'weight', 'bodyId_post'], ascending=[True, False, True], inplace=True)
    conn_df.reset_index(drop=True, inplace=True)

    # Append status column.
    conn_df = conn_df.merge(status_df, left_on='bodyId_post', right_on='bodyId').drop(columns={'bodyId'})

    # Drop non-orphans.
    conn_df.query('status not in @complete_statuses and statusLabel not in @complete_statuses', inplace=True)
    conn_df.rename(columns={'status': 'status_post', 'weight': 'orphan_weight'}, inplace=True)

    # Calculate current output completeness
    completeness_df = fetch_output_completeness(criteria, complete_statuses, client=client)
    completeness_df = completeness_df.rename(columns={'completeness': 'orig_completeness',
                                                      'traced_weight': 'orig_traced_weight',
                                                      'untraced_weight': 'orig_untraced_weight'})

    # Calculate the potential output completeness we would
    # achieve if these orphans became traced, one-by-one.
    conn_df = conn_df.merge(completeness_df, 'left', left_on='bodyId_pre', right_on='bodyId').drop(columns=['bodyId'])
    conn_df['cum_orphan_weight'] = conn_df.groupby('bodyId_pre')['orphan_weight'].cumsum()
    conn_df['cum_completeness'] = conn_df.eval('(orig_traced_weight + cum_orphan_weight) / total_weight')

    return conn_df
