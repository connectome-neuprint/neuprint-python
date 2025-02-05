"""
Miscellaneous utilities for wrangling data from neuprint for various purposes.
"""
import pandas as pd
import numpy as np


def syndist_matrix(syndist, rois=None, syn_columns=['pre', 'post'], flatten_column_index=False):
    """
    Pivot a synapse ROI counts table (one row per body).

    Given a table of synapse ROI distributions as returned by :py:func:`.fetch_neurons()`,
    pivot the ROIs into the columns so the result has one row per body.

    Args:
        syndist:
            DataFrame in the format returned by ``fetch_neurons()[1]``
        rois:
            Optionally filter the input table to process only the listed ROIs.
        syn_columns:
            Optionally process only the given columns of syndist.
        flatten_column_index:
            By default, the result columns will use a MultiIndex ``(orig_col, roi)``,
            e.g. ``('pre', 'LO(R)')``.  If ``flatten_column_index=True``, then the
            output column index is flattened to a plain index with names like ``LO(R)-pre``.
    Returns:
        DataFrame indexed by bodyId and with column count C * R, where C
        is the number of original columns (not counting bodId and roi),
        and R is the number of unique rois in the input.

    Example:

        .. code-block:: ipython

            In [1]: from neuprint import Client, fetch_neurons, syndist_matrix
               ...: c = Client('neuprint.janelia.org', 'hemibrain:v1.2.1')
               ...: bodies = [786989471, 925548084, 1102514975, 1129042596, 1292847181, 5813080979]
               ...: neurons, syndist = fetch_neurons(bodies)
               ...: syndist_matrix(syndist, ['EB', 'FB', 'PB'])
            Out[1]:
                        pre           post
            roi          EB   FB  PB    EB    FB   PB
            bodyId
            786989471     0  110  11     0  1598  157
            925548084     0  542   0     0   977    0
            1102514975    0  236   0     1  1338    0
            1129042596    0  139   0     0  1827    0
            1292847181  916    0   0  1558     0    0
            5813080979  439    0   0   748     0  451
    """
    if rois is not None:
        syndist = syndist.query('roi in @rois')
    if syn_columns is not None and len(syn_columns) > 0:
        syndist = syndist[['bodyId', 'roi', *syn_columns]]

    matrix = syndist.set_index(['bodyId', 'roi']).unstack(fill_value=0)

    if flatten_column_index:
        matrix.columns = [f"{roi}-{prepost}" for (prepost, roi) in matrix.columns.values]

    return matrix


def bilateral_syndist(syndist, bodies=None, rois=None, syn_columns=['pre', 'post']):
    """
    Aggregate synapse counts for corresponding left and right ROIs.

    Given a synapse distribution table as returned by :py:func:`.fetch_neurons()`
    (in its second return value), group corresponding contralateral ROIs
    (suffixed with ``(L)`` and ``(R)``) and aggregate their synapse counts
    into total 'bilateral' counts with the suffix ``(LR)``.

    ROIs without a suffix ``(L)``/``(R)`` will be returned in the output unchanged.

    Args:
        syndist:
            DataFrame in the format returned by ``fetch_neurons()[1]``
        bodies:
            Optionally filter the input table to include only the listed body IDs.
        rois:
            Optionally filter the input table to process only the listed ROIs.
        syn_columns:
            The names of the statistic columns in the input to process.
            Others are ignored.
    Returns:
        DataFrame, similar to the input table but with left/right ROIs aggregated
        and named with a ``(LR)`` suffix.

    Example:

        .. code-block:: ipython

            In [1]: from neuprint import Client, fetch_neurons, bilateral_syndist
               ...: c = Client('neuprint.janelia.org', 'hemibrain:v1.2.1')
               ...: bodies = [786989471, 925548084, 1102514975, 1129042596, 1292847181, 5813080979]
               ...: neurons, syndist = fetch_neurons(bodies)
               ...: bilateral_syndist(syndist, rois=c.primary_rois)
            Out[1]:
                    bodyId      roi  pre  post
            0    786989471  CRE(LR)   77    75
            3    786989471       FB  110  1598
            1    786989471  LAL(LR)    2     2
            14   786989471       PB   11   157
            2    925548084  CRE(LR)    1   203
            22   925548084       FB  542   977
            3    925548084  SMP(LR)    1   171
            4   1102514975  CRE(LR)    2   190
            35  1102514975       EB    0     1
            37  1102514975       FB  236  1338
            5   1102514975  ICL(LR)    0     1
            6   1102514975  LAL(LR)    0     3
            7   1102514975  SMP(LR)    0    74
            8   1102514975  b'L(LR)    0     4
            55  1129042596       FB  139  1827
            9   1129042596  ICL(LR)    0     2
            10  1292847181   BU(LR)    5   143
            67  1292847181       EB  916  1558
            11  1292847181  LAL(LR)    0     1
            77  5813080979       EB  439   748
            82  5813080979       NO  105   451
            86  5813080979       PB    0   451
    """
    if bodies is not None:
        syndist = syndist.query('bodyId in @syndist').copy()

    if rois is not None:
        syndist = syndist.query('roi in @rois').copy()

    if syn_columns is not None and len(syn_columns) > 0:
        syndist = syndist[['bodyId', 'roi', *syn_columns]]

    lateral_matches = syndist['roi'].str.match(r'.*\((R|L)\)')
    syndist_lateral = syndist.loc[lateral_matches].copy()
    syndist_medial = syndist.loc[~lateral_matches].copy()
    syndist_lateral['roi'] = syndist_lateral['roi'].str.slice(0, -3)

    syndist_bilateral = syndist_lateral.groupby(['bodyId', 'roi'], as_index=False).sum()
    syndist_bilateral['roi'] = syndist_bilateral['roi'] + '(LR)'

    syndist_bilateral = pd.concat((syndist_medial, syndist_bilateral))
    syndist_bilateral = syndist_bilateral.sort_values(['bodyId', 'roi'])
    return syndist_bilateral


def assign_sides_in_groups(neurons, syndist, primary_rois=None, min_pre=50, min_post=100, min_bias=0.7):
    """
    Determine which side (left or right) each neuron belongs to,
    according to a few heuristics.

    Assigns a column named 'consensusSide' to the given neurons table.
    The consensusSide is only assigned for neurons with an assigned ``group``,
    and only if every neuron in the group can be assigned a side using
    the same heuristic.

    The neurons are processed in groups (according to the ``group`` column).
    Multiple heuristics are tried:

    - If all neurons in the group have a valid ``somaSide``, then that's used.
    - Otherwise, if all neurons in the group have an instance ending with
      ``_L`` or ``_R``, then that is used.
    - Otherwise, we inspect the pre- and post-synapse counts in ROIs which end
      with ``(L)`` or ``(R)``:

        - If all neurons in the group have significantly more post-synapses
          on one side, then the balance post-synapse is used to assign the
          neuron side.
        - Otherwise, if all neurons in the group have significantly more
          pre-synapses on one side, then that's used.
        - But we do not use either heuristic if there is any disagreement
          on the relative lateral direction in which the neurons in the group
          project.  If some seem to project contralaterally and others seem to
          project ipsilaterally, we do not assign a consensusSide to any neurons
          in the group.

    Args:
        neurons:
            As produced by :py:func:`.fetch_neurons()`
        syndist:
            As produced by :py:func:`.fetch_neurons()`
        primary_rois:
            To avoid double-counting synapses in overlapping ROIs, it is best to
            restrict the syndist table to non-overlapping ROIs only (e.g. primary ROIs).
            Provide the list of such ROIs here, or pre-filter the input yourself.
        min_pre:
            When determining a neuron's side via synapse counts, don't analyze
            pre-synapses in neurons with fewer than ``min_pre`` pre-synapses.
        min_post:
            When determining a neuron's side via synapse counts, don't analyze
            post-synapses in neurons with fewer than ``min_post`` post-synapses.
        min_bias:
            When determining a neuron's side via synapse counts, don't assign a
            consensusSide unless each neuron in the group has a significant fraction
            of its lateral synapses on either the left or right, as specified
            in this argument.  By default, only assign a consensusSide if 70%
            of post-synapses are on one side, or 70% of pre-synapses are on one
            side (not counting synapses in medial ROIs).

    Returns:
        DataFrame, indexed by bodyId, with column ``consensusSide`` (all values
        ``L``, ``R``, or ``None``) and various auxiliary columns which indicate
        how the consensus was determined.
    """
    neurons = neurons.copy()
    neurons.index = neurons['bodyId']

    # According to instance, what side is the neuron on?
    neurons['instanceSide'] = neurons['instance'].astype(str).str.extract('.*_(R|L)(_.*)?')[0]

    # According to the fraction of pre and post, what side is the neuron on?
    if primary_rois is not None:
        syndist = syndist.query('roi in @primary_rois')

    syndist = syndist.copy()
    syndist['roiSide'] = syndist['roi'].str.extract(r'.*\((R|L)\)').fillna('M')
    body_roi_sums = syndist.groupby(['bodyId', 'roiSide'])[['pre', 'post']].sum()
    body_sidecounts = body_roi_sums.unstack(fill_value=0)

    # body_sums = body_roi_sums.groupby('bodyId').sum()
    # body_sidefrac = (body_roi_sums / body_sums).unstack(fill_value=0)
    # body_sidefrac.columns = [f"{prepost}_frac_{side}" for (prepost, side) in body_sidefrac.columns.values]

    neurons['preSide'] = (
        body_sidecounts['pre']
        .query('(L + M + R) >= @min_pre and (L / (L+R) > @min_bias or R / (L+R) > @min_bias)')
        .idxmax(axis=1)
        .replace('M', np.nan)
    )
    neurons['postSide'] = (
        body_sidecounts['post']
        .query('(L + M + R) >= @min_post and (L / (L+R) > @min_bias or R / (L+R) > @min_bias)')
        .idxmax(axis=1)
        .replace('M', np.nan)
    )

    sides = {}
    methods = {}
    for _, df in neurons.groupby('group'):
        # For this function, 'M' is considered null
        if df['somaSide'].isin(['L', 'R']).all():
            sides |= dict(df['somaSide'].items())
            methods |= {body: 'somaSide' for body in df.index}
            continue

        if df['instanceSide'].notnull().all():
            sides |= dict(df['instanceSide'].items())
            methods |= {body: 'instanceSide' for body in df.index}
            continue

        # - Either pre or post must be complete (no NaN).
        if not (df['preSide'].notnull().all() or df['postSide'].notnull().all()):
            continue

        # - If pre and post are both known, then we infer the neuron to be projecting
        #   ipsilaterally or contralaterally, and all neurons in the group who CAN be
        #   assigned a projection direction must agree on the direction of the projection.
        #   But if some cannot be assigned a projection direction and some can, we don't balk.
        has_pre_and_post = df['preSide'].notnull() & df['postSide'].notnull()
        if has_pre_and_post.any():
            is_ipsi = df.loc[has_pre_and_post, 'preSide'] == df.loc[has_pre_and_post, 'postSide']
            if is_ipsi.any() and not is_ipsi.all():
                continue

        # - Prefer the postSide (if available) as the final answer,
        #   since somaSide is usually the postSide (in flies, anyway).
        if df['postSide'].notnull().all():
            sides |= dict(df['postSide'].items())
            methods |= {body: 'postSide' for body in df.index}
        else:
            assert df['preSide'].notnull().all()
            sides |= dict(df['preSide'].items())
            methods |= {body: 'preSide' for body in df.index}

    neurons['consensusSide'] = pd.Series(sides)
    neurons['consensusSide'] = neurons['consensusSide'].replace(np.nan, None)

    neurons['consensusSideMethod'] = pd.Series(methods)
    neurons['consensusSideMethod'] = neurons['consensusSideMethod'].replace(np.nan, None)

    def allnotnull(s):
        return s.notnull().all()

    def allnull(s):
        return s.isnull().all()

    # Sanity check:
    # Every group should EITHER have no consensusSide at all,
    # or should have a consensusSide for every neuron in the group.
    # No groups should have any bodies with a consensusSide
    # unless all bodies in the group have a consensusSide.
    aa = neurons.groupby('group')['consensusSide'].agg([allnotnull, allnull])
    assert (aa['allnull'] ^ aa['allnotnull']).all()

    return neurons[['instanceSide', 'preSide', 'postSide', 'consensusSide', 'consensusSideMethod']]
