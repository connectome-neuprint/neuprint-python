import sys
import copy
from textwrap import dedent

import numpy as np
import pandas as pd

from ..client import inject_client
from ..utils import tqdm, iter_batches
from .neuroncriteria import neuroncriteria_args
from .synapsecriteria import SynapseCriteria
from .mitocriteria import MitoCriteria
from .synapses import fetch_synapse_connections


@inject_client
@neuroncriteria_args('neuron_criteria')
def fetch_mitochondria(neuron_criteria, mito_criteria=None, batch_size=10, *, client=None):
    """
    Fetch mitochondria from a neuron or selection of neurons.

    Args:

        neuron_criteria (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            Determines which bodies from which to fetch mitochondria.

            Note:
                Any ROI criteria specified in this argument does not affect
                which mitochondria are returned, only which bodies are inspected.

        mito_criteria (MitoCriteria):
            Optional. Allows you to filter mitochondria by roi, mitoType, size.
            See :py:class:`.MitoCriteria` for details.

            If the criteria specifies ``primary_only=True`` only primary ROIs will be returned in the results.
            If a mitochondrion does not intersect any primary ROI, it will be listed with an roi of ``None``.
            (Since 'primary' ROIs do not overlap, each mitochondrion will be listed only once.)
            Otherwise, all ROI names will be included in the results.
            In that case, some mitochondria will be listed multiple times -- once per intersecting ROI.
            If a mitochondria does not intersect any ROI, it will be listed with an roi of ``None``.

        batch_size:
            To improve performance and avoid timeouts, the mitochondria for multiple bodies
            will be fetched in batches, where each batch corresponds to N bodies.

        client:
            If not provided, the global default :py:class:`.Client` will be used.

    Returns:

        DataFrame in which each row represent a single synapse.
        If ``primary_only=False`` was specified in ``mito_criteria``, some mitochondria
        may be listed more than once, if they reside in more than one overlapping ROI.

    Example:

        .. code-block:: ipython

            In [1]: from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapses
               ...:
               ...: # Consider only neurons which innervate EB
               ...: nc = NC(type='ExR.*', rois=['EB'])
               ...:
               ...: # But return only large mitos from those neurons that reside in the FB or LAL(R)
               ...: mc = MC(rois=['FB', 'LAL(R)'], size=100_000)
               ...: fetch_mitochondria(nc, mc)
            Out[1]:
                      bodyId mitoType     roi      x      y      z     size          r0         r1        r2
            0     1136865339     dark  LAL(R)  15094  30538  23610   259240  101.586632  31.482559  0.981689
            1     1136865339     dark  LAL(R)  14526  30020  23464   297784   67.174950  36.328964  0.901079
            2     1136865339     dark  LAL(R)  15196  30386  23336   133168   54.907104  25.761894  0.912385
            3     1136865339     dark  LAL(R)  14962  30126  23184   169776   66.780258  27.168915  0.942389
            4     1136865339     dark  LAL(R)  15004  30252  23164   148528   69.316467  24.082989  0.951892
            ...          ...      ...     ...    ...    ...    ...      ...         ...        ...       ...
            2807  1259386264     dark      FB  18926  24632  21046   159184   99.404472  21.919170  0.984487
            2808  1259386264     dark      FB  22162  24474  22486   127968   94.380531  20.547171  0.985971
            2809  1259386264   medium      FB  19322  24198  21952  1110888  116.050323  66.010017  0.954467
            2810  1259386264     dark      FB  19272  23632  21728   428168   87.865768  40.370171  0.944690
            2811  1259386264     dark      FB  19208  23442  21602   141928   53.694149  29.956501  0.919831

            [2812 rows x 10 columns]
    """
    mito_criteria = copy.copy(mito_criteria) or MitoCriteria()
    mito_criteria.matchvar = 'm'
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
        batch_df = _fetch_mitos(batch_criteria, mito_criteria, client)
        if len(batch_df) > 0:
            batch_dfs.append( batch_df )

    if batch_dfs:
        return pd.concat( batch_dfs, ignore_index=True )

    # Return empty results, but with correct dtypes
    dtypes = {
        'bodyId': np.dtype('int64'),
        'mitoType': np.dtype('O'),
        'roi': np.dtype('O'),
        'x': np.dtype('int32'),
        'y': np.dtype('int32'),
        'z': np.dtype('int32'),
        'size': np.dtype('int32'),
        'r0': np.dtype('float32'),
        'r1': np.dtype('float32'),
        'r2': np.dtype('float32'),
    }

    return pd.DataFrame([], columns=dtypes.keys()).astype(dtypes)


def _fetch_mitos(neuron_criteria, mito_criteria, client):
    if mito_criteria.primary_only:
        return_rois = {*client.primary_rois}
    else:
        return_rois = {*client.all_rois}

    # If the user specified rois to filter mitos by, but hasn't specified rois
    # in the NeuronCriteria, add them to the NeuronCriteria to speed up the query.
    if mito_criteria.rois and not neuron_criteria.rois:
        neuron_criteria.rois = {*mito_criteria.rois}
        neuron_criteria.roi_req = 'any'

    # Fetch results
    cypher = dedent(f"""\
        {neuron_criteria.global_with(prefix=8)}
        MATCH (n:{neuron_criteria.label})
        {neuron_criteria.all_conditions('n', prefix=8)}

        MATCH (n)-[:Contains]->(:ElementSet)-[:Contains]->(m:Element {{type: "mitochondrion"}})

        {mito_criteria.condition('n', 'm', prefix=8)}

        RETURN n.bodyId as bodyId,
               m.mitoType as mitoType,
               m.size as size,
               m.location.x as x,
               m.location.y as y,
               m.location.z as z,
               m.r0 as r0,
               m.r1 as r1,
               m.r2 as r2,
               apoc.map.removeKeys(m, ['location', 'type', 'mitoType', 'size', 'r0', 'r1', 'r2']) as mito_info
    """)
    data = client.fetch_custom(cypher, format='json')['data']

    # Assemble DataFrame
    mito_table = []
    for body, mitoType, size, x, y, z, r0, r1, r2, mito_info in data:
        # Exclude non-primary ROIs if necessary
        mito_rois = return_rois & {*mito_info.keys()}
        # Fixme: Filter for the user's ROIs (drop duplicates)
        for roi in mito_rois:
            mito_table.append((body, mitoType, roi, x, y, z, size, r0, r1, r2))

        if not mito_rois:
            mito_table.append((body, mitoType, None, x, y, z, size, r0, r1, r2))

    cols = ['bodyId', 'mitoType', 'roi', 'x', 'y', 'z', 'size', 'r0', 'r1', 'r2']
    mito_df = pd.DataFrame(mito_table, columns=cols)

    # Save RAM with smaller dtypes and interned strings
    mito_df['mitoType'] = mito_df['mitoType'].apply(lambda s: sys.intern(s) if s else s)
    mito_df['roi'] = mito_df['roi'].apply(lambda s: sys.intern(s) if s else s)
    mito_df['x'] = mito_df['x'].astype(np.int32)
    mito_df['y'] = mito_df['y'].astype(np.int32)
    mito_df['z'] = mito_df['z'].astype(np.int32)
    mito_df['size'] = mito_df['size'].astype(np.int32)
    mito_df['r0'] = mito_df['r0'].astype(np.float32)
    mito_df['r1'] = mito_df['r1'].astype(np.float32)
    mito_df['r2'] = mito_df['r2'].astype(np.float32)
    return mito_df


@inject_client
@neuroncriteria_args('neuron_criteria')
def fetch_synapses_and_closest_mitochondria(neuron_criteria, synapse_criteria=None, *, batch_size=10, client=None):
    """
    Fetch a set of synapses from a selection of neurons and also return
    their nearest mitocondria (by path-length within the neuron segment).

    Note:
        Some synapses have no nearby mitochondria, possibly due to
        fragmented segmentation around the synapse point.
        Such synapses ARE NOT RETURNED by this function. They're omitted.

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

        client:
            If not provided, the global default :py:class:`.Client` will be used.

    Returns:

        DataFrame in which each row represent a single synapse,
        along with information about its closest mitochondrion.
        Unless ``primary_only`` was specified, some synapses may be listed more than once,
        if they reside in more than one overlapping ROI.

        The synapse coordinates will be returned in columns ``x,y,z``,
        and the mitochondria centroids will be stored in columns ``mx,my,mz``.

        The ``distance`` column indicates the distance from the synapse coordinate to the
        nearest edge of the mitochondria (not the centroid), as traveled along the neuron
        dendrite (not euclidean distance).  The distance is given in voxel units (e.g. 8nm),
        not nanometers.  See release notes concerning the estimated error of these measurements.

    Example:

        .. code-block:: ipython

            In [1]: from neuprint import fetch_synapses_and_closest_mitochondria, NeuronCriteria as NC, SynapseCriteria as SC
               ...: fetch_synapses_and_closest_mitochondria(NC(type='ExR2'), SC(type='pre'))
            Out[1]:
                      bodyId type     roi      x      y      z  confidence mitoType    distance    size     mx     my     mz          r0         r1        r2
            0     1136865339  pre      EB  25485  22873  19546       0.902   medium  214.053040  410544  25544  23096  19564  105.918625  35.547806  0.969330
            1     1136865339  pre      EB  25985  25652  23472       0.930     dark   19.313709   90048  26008  25646  23490   81.459419  21.493509  0.988575
            2     1136865339  pre  LAL(R)  14938  29149  22604       0.826     dark  856.091736  495208  14874  29686  22096   64.086639  46.906826  0.789570
            3     1136865339  pre      EB  24387  23583  20681       0.945     dark   78.424950  234760  24424  23536  20752   80.774353  29.854616  0.957713
            4     1136865339  pre   BU(R)  16909  25233  17658       0.994     dark  230.588562  215160  16862  25418  17824   42.314690  36.891937  0.628753
            ...          ...  ...     ...    ...    ...    ...         ...      ...         ...     ...    ...    ...    ...         ...        ...       ...
            4508   787762461  pre   BU(R)  16955  26697  17300       0.643     dark  105.765854  176952  16818  26642  17200   91.884338  22.708199  0.975422
            4509   787762461  pre  LAL(R)  15008  28293  25995       0.747     dark  112.967644  446800  15044  28166  26198  176.721512  27.971079  0.992517
            4510   787762461  pre      EB  23468  24073  20882       0.757     dark  248.562714   92536  23400  23852  20760   39.696674  27.490204  0.860198
            4511   787762461  pre   BU(R)  18033  25846  20393       0.829     dark   38.627419  247640  18028  25846  20328   73.585144  29.661413  0.929788
            4512   787762461  pre      EB  22958  24565  20340       0.671     dark  218.104736  120880  23148  24580  20486   39.752777  32.047478  0.821770

            [4513 rows x 16 columns]
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
        batch_df = _fetch_synapses_and_closest_mitochondria(batch_criteria, synapse_criteria, client)
        if len(batch_df) > 0:
            batch_dfs.append( batch_df )

    if batch_dfs:
        return pd.concat( batch_dfs, ignore_index=True )

    # Return empty results, but with correct dtypes
    dtypes = {
        'bodyId': np.dtype('int64'),
        'type': pd.CategoricalDtype(categories=['pre', 'post'], ordered=False),
        'roi': np.dtype('O'),
        'x': np.dtype('int32'),
        'y': np.dtype('int32'),
        'z': np.dtype('int32'),
        'confidence': np.dtype('float32'),
        'mitoType': np.dtype('O'),
        'distance': np.dtype('float32'),
        'mx': np.dtype('int32'),
        'my': np.dtype('int32'),
        'mz': np.dtype('int32'),
        'size': np.dtype('int32'),
        'r0': np.dtype('float32'),
        'r1': np.dtype('float32'),
        'r2': np.dtype('float32'),
    }

    return pd.DataFrame([], columns=dtypes.keys()).astype(dtypes)


def _fetch_synapses_and_closest_mitochondria(neuron_criteria, synapse_criteria, client):

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

    # Fetch results
    cypher = dedent(f"""\
        {neuron_criteria.global_with(prefix=8)}
        MATCH (n:{neuron_criteria.label})
        {neuron_criteria.all_conditions('n', prefix=8)}

        MATCH (n)-[:Contains]->(ss:SynapseSet)-[:Contains]->(s:Synapse)-[c:CloseTo]->(m:Element {{type: "mitochondrion"}})

        {synapse_criteria.condition('n', 's', 'm', 'c', prefix=8)}
        // De-duplicate 's' because 'pre' synapses can appear in more than one SynapseSet
        WITH DISTINCT n, s, m, c

        RETURN n.bodyId as bodyId,
               s.type as type,
               s.confidence as confidence,
               s.location.x as x,
               s.location.y as y,
               s.location.z as z,
               apoc.map.removeKeys(s, ['location', 'confidence', 'type']) as syn_info,
               m.mitoType as mitoType,
               c.distance as distance,
               m.size as size,
               m.location.x as mx,
               m.location.y as my,
               m.location.z as mz,
               m.r0 as r0,
               m.r1 as r1,
               m.r2 as r2
    """)
    data = client.fetch_custom(cypher, format='json')['data']

    # Assemble DataFrame
    syn_table = []
    for body, syn_type, conf, x, y, z, syn_info, mitoType, distance, size, mx, my, mz, r0, r1, r2 in data:
        # Exclude non-primary ROIs if necessary
        syn_rois = return_rois & {*syn_info.keys()}
        # Fixme: Filter for the user's ROIs (drop duplicates)
        for roi in syn_rois:
            syn_table.append((body, syn_type, roi, x, y, z, conf, mitoType, distance, size, mx, my, mz, r0, r1, r2))

        if not syn_rois:
            syn_table.append((body, syn_type, None, x, y, z, conf, mitoType, distance, size, mx, my, mz, r0, r1, r2))

    cols = [
        'bodyId',
        'type', 'roi', 'x', 'y', 'z', 'confidence',
        'mitoType', 'distance', 'size', 'mx', 'my', 'mz', 'r0', 'r1', 'r2'
    ]
    syn_df = pd.DataFrame(syn_table, columns=cols)

    # Save RAM with smaller dtypes and interned strings
    syn_df['type'] = pd.Categorical(syn_df['type'], ['pre', 'post'])
    syn_df['roi'] = syn_df['roi'].apply(lambda s: sys.intern(s) if s else s)
    syn_df['x'] = syn_df['x'].astype(np.int32)
    syn_df['y'] = syn_df['y'].astype(np.int32)
    syn_df['z'] = syn_df['z'].astype(np.int32)
    syn_df['confidence'] = syn_df['confidence'].astype(np.float32)
    syn_df['mitoType'] = syn_df['mitoType'].apply(lambda s: sys.intern(s) if s else s)
    syn_df['distance'] = syn_df['distance'].astype(np.float32)
    syn_df['size'] = syn_df['size'].astype(np.int32)
    syn_df['mx'] = syn_df['mx'].astype(np.int32)
    syn_df['my'] = syn_df['my'].astype(np.int32)
    syn_df['mz'] = syn_df['mz'].astype(np.int32)
    syn_df['r0'] = syn_df['r0'].astype(np.float32)
    syn_df['r1'] = syn_df['r1'].astype(np.float32)
    syn_df['r2'] = syn_df['r2'].astype(np.float32)
    return syn_df


@inject_client
@neuroncriteria_args('source_criteria', 'target_criteria')
def fetch_connection_mitochondria(source_criteria, target_criteria, synapse_criteria=None, min_total_weight=1, *, client=None):
    """
    For a given set of source neurons and target neurons, find all
    synapse-level connections between the sources and targets, along
    with the nearest mitochondrion on the pre-synaptic side and the
    post-synaptic side.

    Returns a table similar to :py:func:`fetch_synapse_connections()`, but with
    extra ``_pre`` and ``_post`` columns to describe the nearest mitochondria
    to the pre/post synapse in the connection.
    If a given synapse has no nearby mitochondrion, the corresponding
    mito columns will be populated with ``NaN`` values. (This is typically
    much more likely to occur on the post-synaptic side than the pre-synaptic side.)

    Arguments are the same as :py:func:`fetch_synapse_connections()`

    Note:
        This function does not employ a custom cypher query to minimize the
        data fetched from the server. Instead, it makes multiple calls to the
        server and merges the results on the client.

    Args:
        source_criteria (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            Criteria to by which to filter source (pre-synaptic) neurons.
            If omitted, all Neurons will be considered as possible sources.

            Note:
                Any ROI criteria specified in this argument does not affect
                which synapses are returned, only which bodies are inspected.

        target_criteria (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
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

        client:
            If not provided, the global default :py:class:`.Client` will be used.

    """
    SC = SynapseCriteria

    # Fetch the synapses that connect sources and targets
    # (subject to min_total_weight)
    conn = fetch_synapse_connections(source_criteria, target_criteria, synapse_criteria, min_total_weight, batch_size=10)

    output_bodies = conn['bodyId_pre'].unique()
    output_mito = fetch_synapses_and_closest_mitochondria(output_bodies, SC(type='pre', client=client), batch_size=1)
    output_mito = output_mito[[*'xyz', 'mitoType', 'distance', 'size', 'mx', 'my', 'mz']]
    output_mito = output_mito.rename(columns={'size': 'mitoSize'})

    input_bodies = conn['bodyId_post'].unique()
    input_mito = fetch_synapses_and_closest_mitochondria(input_bodies, SC(type='post', client=client), batch_size=1)
    input_mito = input_mito[[*'xyz', 'mitoType', 'distance', 'size', 'mx', 'my', 'mz']]
    input_mito = input_mito.rename(columns={'size': 'mitoSize'})

    # This double-merge will add _pre and _post columns for the mito fields
    conn_with_mito = conn
    conn_with_mito = conn_with_mito.merge(output_mito,
                                          'left',
                                          left_on=['x_pre', 'y_pre', 'z_pre'],
                                          right_on=['x', 'y', 'z']).drop(columns=[*'xyz'])

    conn_with_mito = conn_with_mito.merge(input_mito,
                                          'left',
                                          left_on=['x_post', 'y_post', 'z_post'],
                                          right_on=['x', 'y', 'z'],
                                          suffixes=['_pre', '_post']).drop(columns=[*'xyz'])
    return conn_with_mito
