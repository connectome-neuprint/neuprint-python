import sys
import copy
from textwrap import dedent

import numpy as np
import pandas as pd

from ..client import inject_client
from ..utils import tqdm, iter_batches
from .neuroncriteria import NeuronCriteria, neuroncriteria_args
from .synapsecriteria import SynapseCriteria
from .connectivity import fetch_adjacencies


@inject_client
@neuroncriteria_args('neuron_criteria')
def fetch_synapses(neuron_criteria, synapse_criteria=None, batch_size=10, *, nt=None, client=None):
    """
    Fetch synapses from a neuron or selection of neurons.

    Args:

        neuron_criteria (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            Determines which bodies to fetch synapses for.

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

        batch_size:
            To improve performance and avoid timeouts, the synapses for multiple bodies
            will be fetched in batches, where each batch corresponds to N bodies.
            This argument sets the batch size N.

        nt (None (default), 'max', or 'all'):
            Optional. Retrieves neurotransmitter information for each "pre" synapse.

            If None, no neurotransmitter information is retrieved.
            If 'max', the most probable neurotransmitter for each synapse is returned in a column
            named "nt" and a column "ntProb" indicating the probability associated with it.
            If 'all', probabilities for all neurotransmitters are returned in named columns.

            If no neurotransmitter information is available, then setting nt to 'max' or 'all'
            will raise an error.

        client:
            If not provided, the global default :py:class:`.Client` will be used.

    Returns:

        DataFrame in which each row represent a single synapse.
        Unless ``primary_only`` was specified, some synapses may be listed more than once,
        if they reside in more than one overlapping ROI.

    Example:

        .. code-block:: ipython

            In [1]: from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapses
               ...: fetch_synapses(NC(type='ADL.*', rois=['FB']),
               ...:                SC(rois=['LH(R)', 'SIP(R)'], primary_only=True))
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
    neuron_criteria.matchvar = 'n'
    q = f"""
        {neuron_criteria.global_with(prefix=8)}
        MATCH (n:{neuron_criteria.label})
        {neuron_criteria.all_conditions(prefix=8)}
        RETURN n.bodyId as bodyId
    """
    bodies = client.fetch_custom(q)['bodyId'].values

    batch_dfs = []
    for batch_bodies in tqdm(iter_batches(bodies, batch_size), disable=not client.progress):
        batch_criteria = copy.copy(neuron_criteria)
        batch_criteria.bodyId = batch_bodies
        batch_df = _fetch_synapses(batch_criteria, synapse_criteria, nt, client)
        if len(batch_df) > 0:
            batch_dfs.append( batch_df )

    if batch_dfs:
        return pd.concat( batch_dfs, ignore_index=True )

    # Return empty results, but with correct dtypes
    dtypes = {'bodyId': np.dtype('int64'),
              'type': pd.CategoricalDtype(categories=['pre', 'post'], ordered=False),
              'roi': np.dtype('O'),
              'x': np.dtype('int32'),
              'y': np.dtype('int32'),
              'z': np.dtype('int32'),
              'confidence': np.dtype('float32')}

    return pd.DataFrame([], columns=dtypes.keys()).astype(dtypes)


def _fetch_synapses(neuron_criteria, synapse_criteria, nt, client):
    neuron_criteria.matchvar = 'n'

    if synapse_criteria is None:
        synapse_criteria = SynapseCriteria(client=client)

    if synapse_criteria.primary_only:
        return_rois = {*client.primary_rois}
    else:
        return_rois = {*client.all_rois}

    # If the user specified rois to filter synapses by, but hasn't specified rois
    # in the NeuronCriteria, add them to the NeuronCriteria to speed up the query.
    if synapse_criteria.rois and not neuron_criteria.rois:
        neuron_criteria.rois = {*synapse_criteria.rois}
        neuron_criteria.roi_req = 'any'

    # Neurotransmitters vary by dataset; get the names and dynamically
    #   insert the column names into the cypher query.
    synapse_nt_prop_names = []
    if nt:
        synapse_nt_prop_names = client.fetch_synapse_nt_keys()
        if not synapse_nt_prop_names:
            raise RuntimeError(
                "Can't return synapse neurotransmitter properties: "
                "No neurotransmitter properties found in the database."
            )

    # Fetch results
    cypher = dedent(f"""\
        {neuron_criteria.global_with(prefix=8)}
        MATCH (n:{neuron_criteria.label})
        {neuron_criteria.all_conditions('n', prefix=8)}

        MATCH (n)-[:Contains]->(ss:SynapseSet),
              (ss)-[:Contains]->(s:Synapse)

        {synapse_criteria.condition('n', 's', prefix=8)}
        // De-duplicate 's' because 'pre' synapses can appear in more than one SynapseSet
        WITH DISTINCT n, s
        
        // Extract properties as rows
        RETURN n.bodyId as bodyId,
               s.type as type,
               s.confidence as confidence,
               s.location.x as x,
               s.location.y as y,
               s.location.z as z,
               apoc.map.removeKeys(s, ['location', 'confidence', 'type']) as syn_info
    """)

    if nt and synapse_nt_prop_names:
        cypher = cypher[:-1] + ',\n'
        cypher += _neurotransmitter_return_clause(synapse_nt_prop_names, prefix=15)

    data = client.fetch_custom(cypher, format='json')['data']

    # Assemble DataFrame
    syn_table = []
    cleaned_nt_prop_names = [_clean_nt_name(name) for name in synapse_nt_prop_names]
    for body, syn_type, conf, x, y, z, syn_info, *nt_probs in data:

        nt_info = _process_nt_probabilities(nt, nt_probs, cleaned_nt_prop_names)

        # Exclude non-primary ROIs if necessary
        syn_rois = return_rois & {*syn_info.keys()}
        # Fixme: Filter for the user's ROIs (drop duplicates)
        for roi in syn_rois:
            syn_table.append((body, syn_type, roi, x, y, z, conf) + nt_info)

        if not syn_rois:
            syn_table.append((body, syn_type, None, x, y, z, conf) + nt_info)

    synapse_columns = ['bodyId', 'type', 'roi', 'x', 'y', 'z', 'confidence']
    if nt == "max":
        synapse_columns.extend(['nt', 'ntProb'])
    elif nt == "all":
        synapse_columns.extend(cleaned_nt_prop_names)

    syn_df = pd.DataFrame(syn_table, columns=synapse_columns)

    # Save RAM with smaller dtypes and interned strings
    syn_df['type'] = pd.Categorical(syn_df['type'], ['pre', 'post'])
    syn_df['roi'] = syn_df['roi'].apply(lambda s: sys.intern(s) if s else s)
    syn_df['x'] = syn_df['x'].astype(np.int32)
    syn_df['y'] = syn_df['y'].astype(np.int32)
    syn_df['z'] = syn_df['z'].astype(np.int32)
    syn_df['confidence'] = syn_df['confidence'].astype(np.float32)

    # nt columns types
    if nt == 'all':
        for column in cleaned_nt_prop_names:
            syn_df[column] = syn_df[column].astype(np.float32)
    elif nt == 'max':
        syn_df['nt'] = pd.Categorical(syn_df['nt'], cleaned_nt_prop_names)
        syn_df['ntProb'] = syn_df['ntProb'].astype(np.float32)

    return syn_df


@inject_client
@neuroncriteria_args('neuron_criteria')
def fetch_mean_synapses(neuron_criteria, synapse_criteria=None, batch_size=100, *, by_roi=True, client=None):
    """
    Fetch average synapse position and confidence for each neuron in set of neurons.

    Args:

        neuron_criteria (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            Determines which bodies to fetch synapses for.

            Note:
                Any ROI criteria specified in this argument does not affect
                which synapses are returned, only which bodies are inspected.

        synapse_criteria (SynapseCriteria):
            Optional. Allows you to filter synapses by roi, type, confidence.
            See :py:class:`.SynapseCriteria` for details.

            If the criteria specifies ``primary_only=True`` only primary ROIs will be returned in the results.
            If a synapse does not intersect any primary ROI, it will be ignored by this function.

        batch_size:
            To improve performance and avoid timeouts, the synapses for multiple bodies
            will be processed in batches, where each batch corresponds to N bodies.
            This argument sets the batch size N.

        by_roi:
            If ``by_roi=True`` (the default), return separate rows for each ROI in each neuron.
            If ``by_roi=False``, then return the mean for the entire neuron in each case,
            without respect to ROI.

            Note:

                Even if ``by_roi=False``, the set of averaged synapses will be restricted to the
                ROIs listed in your `synapse_critera` (if any).

        client:
            If not provided, the global default :py:class:`.Client` will be used.

    Returns:

        DataFrame in which each row contains the average synapse position and average
        confidence for a particular body, ROI, and synapse type (pre/post).

    Example:

        .. code-block:: ipython

            In [1]: from neuprint import fetch_mean_synapses, SynapseCriteria as SC
               ...: fetch_mean_synapses('LC10', SC(type='pre', rois=['LO(R)', 'AOTU(R)']))
            Out[1]:
                    bodyId      roi type  count             x             y             z  confidence
            0    1017448980  AOTU(R)  pre    141  10691.737305  30304.355469  15508.099609    0.956000
            1    1017448980    LO(R)  pre      2   8530.500000  18898.000000  34857.000000    0.954500
            2    1017090133  AOTU(R)  pre     49  10560.469727  31032.041016  14548.550781    0.956286
            3    1017090133    LO(R)  pre     52   3737.115479  19489.923828  29530.000000    0.949673
            4    1017094185  AOTU(R)  pre    157  10922.248047  30326.693359  15349.484375    0.959185
            ..          ...      ...  ...    ...           ...           ...           ...         ...
            772  1262602271    LO(R)  pre      3   5178.666504  14584.666992  23227.333984    0.957333
            773  1291573712  AOTU(R)  pre    273  11285.106445  31097.925781  15987.640625    0.954418
            774  1291573712    LO(R)  pre      3   6482.000000  13845.333008  25962.666016    0.936333
            775  5812993252  AOTU(R)  pre     17  11110.706055  30892.470703  17220.175781    0.960471
            776  5812993252    LO(R)  pre     39   7758.897461  11723.154297  28176.794922    0.943128
    """
    assert by_roi in (True, False), "by_roi should be boolean"

    neuron_criteria.matchvar = 'n'
    q = f"""
        {neuron_criteria.global_with(prefix=8)}
        MATCH (n:{neuron_criteria.label})
        {neuron_criteria.all_conditions(prefix=8)}
        RETURN n.bodyId as bodyId
    """
    bodies = client.fetch_custom(q)['bodyId'].values

    batch_dfs = []
    for batch_bodies in tqdm(iter_batches(bodies, batch_size), disable=not client.progress):
        batch_criteria = copy.copy(neuron_criteria)
        batch_criteria.bodyId = batch_bodies
        if by_roi:
            batch_df = _fetch_mean_synapses_per_roi(batch_criteria, synapse_criteria, client)
        else:
            batch_df = _fetch_mean_synapses_per_whole_neuron(batch_criteria, synapse_criteria, client)

        if len(batch_df) > 0:
            batch_dfs.append( batch_df )

    if batch_dfs:
        return pd.concat( batch_dfs, ignore_index=True )

    # Return empty results, but with correct dtypes
    dtypes = {'bodyId': np.dtype('int64'),
              'type': pd.CategoricalDtype(categories=['pre', 'post'], ordered=False),
              'roi': np.dtype('O'),
              'count': np.dtype('int32'),
              'x': np.dtype('float32'),
              'y': np.dtype('float32'),
              'z': np.dtype('float32')}

    if not by_roi:
        del dtypes['roi']

    return pd.DataFrame([], columns=dtypes.keys()).astype(dtypes)


def _fetch_mean_synapses_per_roi(neuron_criteria, synapse_criteria, client):
    neuron_criteria.matchvar = 'n'

    if synapse_criteria is None:
        synapse_criteria = SynapseCriteria(client=client)

    if synapse_criteria.rois:
        rois = synapse_criteria.rois
    elif synapse_criteria.primary_only:
        rois = client.primary_rois
    else:
        rois = client.all_rois

    # If the user specified rois to filter synapses by, but hasn't specified rois
    # in the NeuronCriteria, add them to the NeuronCriteria to speed up the query.
    if synapse_criteria.rois and not neuron_criteria.rois:
        neuron_criteria.rois = {*synapse_criteria.rois}
        neuron_criteria.roi_req = 'any'

    # Fetch results
    cypher = dedent(f"""\
        WITH {rois} as rois
        {neuron_criteria.global_with('rois', prefix=8)}
        MATCH (n:{neuron_criteria.label})
        {neuron_criteria.all_conditions('n', prefix=8)}

        MATCH (n)-[:Contains]->(ss:SynapseSet),
              (ss)-[:Contains]->(s:Synapse)

        {synapse_criteria.condition('rois', 'n', 's', prefix=8)}
        // De-duplicate 's' because 'pre' synapses can appear in more than one SynapseSet
        WITH DISTINCT rois, n, s

        // Extract properties as rows
        UNWIND KEYS(s) as roi

        // Filter for the ROIs we need
        WITH rois, n, s, roi
        WHERE roi in rois

        RETURN n.bodyId as bodyId,
               roi,
               s.type as type,
               count(s) as count,
               avg(s.location.x) as x,
               avg(s.location.y) as y,
               avg(s.location.z) as z,
               avg(s.confidence) as confidence
    """)
    syn_df = client.fetch_custom(cypher)

    # Save RAM with smaller dtypes and interned strings
    syn_df['type'] = pd.Categorical(syn_df['type'], ['pre', 'post'])
    syn_df['roi'] = syn_df['roi'].apply(lambda s: sys.intern(s) if s else s)
    syn_df['count'] = syn_df['count'].astype(np.int32)
    syn_df['x'] = syn_df['x'].astype(np.float32)
    syn_df['y'] = syn_df['y'].astype(np.float32)
    syn_df['z'] = syn_df['z'].astype(np.float32)
    syn_df['confidence'] = syn_df['confidence'].astype(np.float32)

    return syn_df


def _fetch_mean_synapses_per_whole_neuron(neuron_criteria, synapse_criteria, client):
    neuron_criteria.matchvar = 'n'

    if synapse_criteria is None:
        synapse_criteria = SynapseCriteria(client=client)

    # If the user specified rois to filter synapses by, but hasn't specified rois
    # in the NeuronCriteria, add them to the NeuronCriteria to speed up the query.
    if synapse_criteria.rois and not neuron_criteria.rois:
        neuron_criteria.rois = {*synapse_criteria.rois}
        neuron_criteria.roi_req = 'any'

    # Fetch results
    cypher = dedent(f"""\
        MATCH (n:{neuron_criteria.label})
        {neuron_criteria.all_conditions('n', prefix=8)}

        MATCH (n)-[:Contains]->(ss:SynapseSet),
              (ss)-[:Contains]->(s:Synapse)

        {synapse_criteria.condition('n', 's', prefix=8)}
        // De-duplicate 's' because 'pre' synapses can appear in more than one SynapseSet
        WITH DISTINCT n, s

        RETURN n.bodyId as bodyId,
               s.type as type,
               count(s) as count,
               avg(s.location.x) as x,
               avg(s.location.y) as y,
               avg(s.location.z) as z,
               avg(s.confidence) as confidence
    """)
    syn_df = client.fetch_custom(cypher)

    # Save RAM with smaller dtypes and interned strings
    syn_df['type'] = pd.Categorical(syn_df['type'], ['pre', 'post'])
    syn_df['count'] = syn_df['count'].astype(np.int32)
    syn_df['x'] = syn_df['x'].astype(np.float32)
    syn_df['y'] = syn_df['y'].astype(np.float32)
    syn_df['z'] = syn_df['z'].astype(np.float32)
    syn_df['confidence'] = syn_df['confidence'].astype(np.float32)

    return syn_df


@inject_client
@neuroncriteria_args('source_criteria', 'target_criteria')
def fetch_synapse_connections(source_criteria=None, target_criteria=None, synapse_criteria=None, min_total_weight=1, batch_size=10_000, *, nt=None, client=None):
    """
    Fetch synaptic-level connections between source and target neurons.

    Note:
        Use this function if you need information about individual synapse connections,
        such as their exact positions or confidence scores.
        If you're just interested in aggregate neuron-to-neuron connection info
        (including connection strengths and ROI intersections), see
        :py:func:`fetch_simple_connections()` and :py:func:`fetch_adjacencies()`,
        which are faster and have more condensed outputs than this function.

    Note:
        If you experience timeouts while running this function,
        try reducing the ``batch_size`` setting.

    Args:

        source_criteria (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            Criteria to by which to filter source (pre-synaptic) neurons.
            If omitted, all Neurons will be considered as possible sources.

            Note:
                Any ROI criteria specified in this argument does not affect
                which synapses are returned, only which bodies are inspected.

        target_criteria (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            Criteria to by which to filter target (post-synaptic) neurons.
            If omitted, all Neurons will be considered as possible targets.

            Note:
                Any ROI criteria specified in this argument does not affect
                which synapses are returned, only which bodies are inspected.

        synapse_criteria (SynapseCriteria):
            Optional. Allows you to filter synapses by roi or confidence.
            The same criteria is used to filter both ``pre`` and ``post`` sides
            of the connection, except for the ``rois`` -- see note below.
            By default, ``SynapseCriteria(primary_only=True)`` is used,
            with no additional filters.

            If ``primary_only`` is specified in the criteria, then the resulting
            ``roi_pre`` and ``roi_post`` columns will contain a single
            string (or ``None``) in every row.

            Otherwise, the roi columns will contain a list of ROIs for every row.
            (Primary ROIs do not overlap, so every synapse resides in only one
            (or zero) primary ROI.)
            See :py:class:`.SynapseCriteria` for details.

            Note:
                Any ``rois`` specified in your ``synapse_criteria`` will be used to filter
                the target (post-synaptic) side of the synapse connection, but not
                the pre-synaptic side.  So in the rare cases where the pre and post synapses
                reside on different sides of an ROI boundary, the connection is associated
                with the post-synaptic ROI.

                That's consistent with neuprint's conventions for ROI assignment in the
                neuron-to-neuron ``ConnectsTo:`` relationship, and thus ensures that this function
                returns synapse counts that are consistent with :py:func:`.fetch_adjacencies()`.


        min_total_weight:
            If the total weight of the connection between two bodies is not at least
            this strong, don't include the synapses for that connection in the results.

            Note:
                This filters for total connection weight, regardless of the weight
                within any particular ROI.  So, if your ``SynapseCriteria`` limits
                results to a particular ROI, but two bodies connect in multiple ROIs,
                then the number of synapses returned for the two bodies may appear to
                be less than ``min_total_weight``. That's because you filtered out
                the synapses in other ROIs.

        batch_size:
            To avoid timeouts and improve performance, the synapse connections
            will be fetched in batches.  The batching strategy is to process
            each body one at a time, and if it has lots of connection partners,
            split the request across several batches to avoid timeouts that
            could arise from a large request.
            This argument specifies the maximum size of each batch in the inner loop.
            Larger batches are more efficient to fetch, but increase the likelihood
            of timeouts.

        neurotransmitters (None (default), 'max', or 'all'):
            Optional. Retrieves neurotransmitter information for each "pre" synapse.

            If None, no neurotransmitter information is retrieved.
            If 'max', the most probable neurotransmitter for each presynapse is returned in a
            column named "nt" along with a column "ntProb" indicating the probability associated with it.
            If 'all', probabilities for all neurotransmitters are returned in named columns.

            If no neurotransmitter information is available, then setting nt to 'max' or 'all'
            will raise an error.

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

            In [1]: from neuprint import fetch_synapse_connections, SynapseCriteria as SC
               ...: fetch_synapse_connections(792368888, None, SC(rois=['PED(R)', 'SMP(R)'], primary_only=True))
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
    assert source_criteria is not None or target_criteria is not None, \
        "Please specify either source or target search criteria (or both)."

    if synapse_criteria is None:
        synapse_criteria = SynapseCriteria(client=client)

    def prepare_nc(nc, matchvar):
        nc.matchvar = matchvar

        # If the user specified rois to filter synapses by, but hasn't specified rois
        # in the NeuronCriteria, add them to the NeuronCriteria to speed up the query.
        if synapse_criteria.rois and not nc.rois:
            nc.rois = {*synapse_criteria.rois}
            nc.roi_req = 'any'

        return nc

    source_criteria = prepare_nc(source_criteria, 'n')
    target_criteria = prepare_nc(target_criteria, 'm')

    # Fetch the list of neuron-neuron pairs in advance so we can break into batches.
    _neuron_df, roi_conn_df = fetch_adjacencies( source_criteria,
                                                 target_criteria,
                                                 synapse_criteria.rois,
                                                 min(1, min_total_weight),
                                                 min_total_weight,
                                                 include_nonprimary=not synapse_criteria.primary_only,
                                                 properties=[],
                                                 client=client )

    if len(roi_conn_df) == 0:
        # Return empty dataframe, but with the correct dtypes
        dtypes = {
            'bodyId_pre': np.dtype('int64'),
            'bodyId_post': np.dtype('int64'),
            'roi_pre': np.dtype('O'),
            'roi_post': np.dtype('O'),
            'x_pre': np.dtype('int32'),
            'y_pre': np.dtype('int32'),
            'z_pre': np.dtype('int32'),
            'x_post': np.dtype('int32'),
            'y_post': np.dtype('int32'),
            'z_post': np.dtype('int32'),
            'confidence_pre': np.dtype('float32'),
            'confidence_post': np.dtype('float32')
        }
        return pd.DataFrame([], columns=dtypes.keys()).astype(dtypes)

    conn_df = (roi_conn_df.drop_duplicates(['bodyId_pre', 'bodyId_post'])
                          .sort_values(['bodyId_pre', 'bodyId_post']))

    # Pick either 'pre' or 'post' column to process bodies one at a time.
    # Then batch across the connections in an inner loop.
    num_pre = conn_df['bodyId_pre'].nunique()
    num_post = conn_df['bodyId_post'].nunique()
    if num_pre < num_post:
        grouping_col = 'bodyId_pre'
    else:
        grouping_col = 'bodyId_post'

    syn_dfs = []
    with tqdm(total=roi_conn_df['weight'].sum(), disable=not client.progress) as progress:
        for _, group_df in conn_df.groupby(grouping_col):
            batches = iter_batches(group_df, batch_size)
            for batch_df in tqdm(batches, leave=False, disable=not client.progress):
                src_crit = copy.copy(source_criteria)
                tgt_crit = copy.copy(target_criteria)

                if grouping_col == 'bodyId_pre':
                    assert batch_df['bodyId_pre'].nunique() == 1
                    src_crit.bodyId = batch_df['bodyId_pre'].unique()
                    # Filter target criteria further only if connections
                    # are being fetched in multiple batches.
                    # Otherwise, the explicit body list is unnecessary and slows down the query.
                    if len(batches) > 1:
                        tgt_crit.bodyId = batch_df['bodyId_post'].unique()
                else:
                    assert batch_df['bodyId_post'].nunique() == 1
                    tgt_crit.bodyId = batch_df['bodyId_post'].unique()
                    # Filter source criteria further only if connections
                    # are being fetched in multiple batches.
                    # Otherwise, the explicit body list is unnecessary and slows down the query.
                    if len(batches) > 1:
                        src_crit.bodyId = batch_df['bodyId_pre'].unique()

                batch_syn_df = _fetch_synapse_connections( src_crit,
                                                           tgt_crit,
                                                           synapse_criteria,
                                                           min_total_weight,
                                                           nt=nt,
                                                           client=client)
                if len(batch_syn_df) > 0:
                    syn_dfs.append(batch_syn_df)
                progress.update(len(batch_syn_df))

    syn_df = pd.concat(syn_dfs, ignore_index=True)
    hashable_cols = [col for col, dtype in syn_df.dtypes.items() if dtype != object]
    assert syn_df.duplicated(hashable_cols).sum() == 0, \
        "Somehow obtained duplicate synapse-synapse connections!"
    return syn_df


def _fetch_synapse_connections(source_criteria, target_criteria, synapse_criteria, min_total_weight, nt, client):
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

    # We apply ROI filtering to the PSD side only,
    # for consistency with the way neuprint assigns ROIs to
    # connections in the neuron ``ConnectsTo:`` relationship.
    # That way, this function and ``fetch_adjacencies()`` return consistent results.
    source_syn_crit.rois = None

    criteria_globals = [
        *source_criteria.global_vars().keys(),
        *target_criteria.global_vars().keys()
    ]
    combined_conditions = NeuronCriteria.combined_conditions(
        [source_criteria, target_criteria],
        ['n', 'e', 'm', 'ns', 'ms', *criteria_globals],
        prefix=8)

    # Neurotransmitters vary by dataset; get the names and dynamically
    #   insert the column names into the cypher query.
    synapse_nt_prop_names = []
    if nt:
        synapse_nt_prop_names = client.fetch_synapse_nt_keys()
        if not synapse_nt_prop_names:
            raise RuntimeError(
                "Can't return synapse neurotransmitter properties: "
                "No neurotransmitter properties found in the database."
            )

    # Fetch results
    cypher = dedent(f"""\
        {NeuronCriteria.combined_global_with((source_criteria, target_criteria), prefix=8)}
        MATCH (n:{source_criteria.label})-[e:ConnectsTo]->(m:{target_criteria.label}),
              (n)-[:Contains]->(nss:SynapseSet)-[:ConnectsTo]->(mss:SynapseSet)<-[:Contains]-(m),
              (nss)-[:Contains]->(ns:Synapse)-[:SynapsesTo]->(ms:Synapse)<-[:Contains]-(mss)

        {combined_conditions}

        // Note: Semantically, the word 'DISTINCT' is unnecessary here,
        // but its presence makes this query run faster.
        WITH DISTINCT n, m, ns, ms, e
        WHERE e.weight >= {min_total_weight}

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

    if nt and synapse_nt_prop_names:
        cypher = cypher[:-1] + ',\n'
        cypher += _neurotransmitter_return_clause(synapse_nt_prop_names, prefix=15, matchvar='ns')

    data = client.fetch_custom(cypher, format='json')['data']

    # Assemble DataFrame
    syn_table = []
    cleaned_nt_prop_names = [_clean_nt_name(name) for name in synapse_nt_prop_names]
    for bodyId_pre, bodyId_post, ux, uy, uz, dx, dy, dz, up_conf, dn_conf, info_pre, info_post, *nt_probs in data:

        nt_info = _process_nt_probabilities(nt, nt_probs, cleaned_nt_prop_names)

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

        syn_table.append((bodyId_pre, bodyId_post, pre_rois, post_rois, ux, uy, uz, dx, dy, dz, up_conf, dn_conf) + nt_info)

    synapse_columns = ['bodyId_pre', 'bodyId_post', 'roi_pre', 'roi_post',
        'x_pre', 'y_pre', 'z_pre', 'x_post', 'y_post', 'z_post', 'confidence_pre', 'confidence_post']
    if nt == "max":
        synapse_columns.extend(['nt', 'ntProb'])
    elif nt == "all":
        synapse_columns.extend(cleaned_nt_prop_names)

    syn_df = pd.DataFrame(syn_table, columns=synapse_columns)

    syn_df['bodyId_pre'] = syn_df['bodyId_pre'].astype(np.int64)
    syn_df['bodyId_post'] = syn_df['bodyId_post'].astype(np.int64)

    # Save RAM with smaller dtypes
    syn_df['x_pre'] = syn_df['x_pre'].astype(np.int32)
    syn_df['y_pre'] = syn_df['y_pre'].astype(np.int32)
    syn_df['z_pre'] = syn_df['z_pre'].astype(np.int32)
    syn_df['x_post'] = syn_df['x_post'].astype(np.int32)
    syn_df['y_post'] = syn_df['y_post'].astype(np.int32)
    syn_df['z_post'] = syn_df['z_post'].astype(np.int32)
    syn_df['confidence_pre'] = syn_df['confidence_pre'].astype(np.float32)
    syn_df['confidence_post'] = syn_df['confidence_post'].astype(np.float32)

    # nt columns types
    if nt == 'all':
        for column in cleaned_nt_prop_names:
            syn_df[column] = syn_df[column].astype(np.float32)
    elif nt == 'max':
        syn_df['nt'] = pd.Categorical(syn_df['nt'], cleaned_nt_prop_names)
        syn_df['ntProb'] = syn_df['ntProb'].astype(np.float32)

    return syn_df

# a few common routines used when working with neurotransmitter info

def _clean_nt_name(name):
    """
    given a neurotransmitter probability name as stored in the database,
    return the name without the 'nt' and 'Prob' parts, lowercase
    """
    return name.replace('nt', '').replace('Prob', '').lower()

def _max_nt(nt_prop_names, nt_probs):
    """
    Return the name of the neurotransmitter with the highest
    probability and the corresponding probability value.

    input: list of nt property names and list of probabilities, in the same order
    """

    # if any probs are None, they all are
    if not nt_probs or nt_probs[0] is None:
        return (None, np.nan)

    max_prob = max(nt_probs)
    nt_name = nt_prop_names[nt_probs.index(max_prob)]
    return (nt_name, max_prob)

def _neurotransmitter_return_clause(synapse_nt_prop_names, prefix="", matchvar='s'):
    """
    Returns the section of the RETURN clause of the cypher query for
    all the neurotransmitter probabilities.

    input: list of the property names; a prefix to indent the query (if integer,
        that many spaces); and the match variable to use
    """
    if isinstance(prefix, int):
        prefix = ' ' * prefix
    if synapse_nt_prop_names:
        return ",\n".join(f"{prefix}{matchvar}.{name} as {name}" for name in synapse_nt_prop_names)
    else:
        return ""


def _process_nt_probabilities(nt, nt_probs, cleaned_nt_prop_names):
    """
    Process neurotransmitter probabilities based on the requested type.

    Args:
        nt: 'max', 'all', or None.
        nt_probs: List of neurotransmitter probabilities.
        cleaned_nt_prop_names: List of cleaned neurotransmitter names.

    Returns:
        Processed neurotransmitter information.
    """
    match nt:
        case None:
            return ()
        case 'max':
            return _max_nt(cleaned_nt_prop_names, nt_probs)
        case 'all':
            return tuple(nt_probs)
        case _:
            raise ValueError(f"Invalid option for nt: {nt}. Use None, 'max', or 'all'.")
