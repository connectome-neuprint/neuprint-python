import pandas as pd
import ujson

from textwrap import indent
from ..client import inject_client
from ..utils import compile_columns
from .neuroncriteria import neuroncriteria_args

# Core set of columns
CORE_NEURON_COLS = ['bodyId', 'instance', 'type',
                    'pre', 'post', 'downstream', 'upstream', 'mito', 'size',
                    'status', 'cropped', 'statusLabel',
                    'cellBodyFiber',
                    'somaRadius', 'somaLocation',
                    'inputRois', 'outputRois', 'roiInfo']


@inject_client
@neuroncriteria_args('criteria')
def fetch_neurons(criteria, *, client=None):
    """
    Return properties and per-ROI synapse counts for a set of neurons.

    Searches for a set of Neurons (or Segments) that match the given :py:class:`.NeuronCriteria`.
    Returns their properties, including the distibution of their synapses in all brain regions.

    This implements a superset of the features on the Neuprint Explorer `Find Neurons`_ page.

    Returns data in the the same format as :py:func:`fetch_custom_neurons()`,
    but doesn't require you to write cypher.

    .. _Find Neurons: https://neuprint.janelia.org/?dataset=hemibrain%3Av1.2.1&qt=findneurons&q=1

    Args:
        criteria (bodyId(s), type/instance, or :py:class:`.NeuronCriteria`):
            Only Neurons which satisfy all components of the given criteria are returned.

        client:
            If not provided, the global default :py:class:`.Client` will be used.

    Returns:
        Two DataFrames.
        ``(neurons_df, roi_counts_df)``

        In ``neurons_df``, all available ``:Neuron`` columns are returned, with the following changes:

            - ROI boolean columns are removed
            - ``roiInfo`` is parsed as json data
            - coordinates (such as ``somaLocation``) are provided as a list ``[x, y, z]``
            - New columns ``input_rois`` and ``output_rois`` contain lists of each neuron's ROIs.

        In ``roi_counts_df``, the ``roiInfo`` has been loadded into a table
        of per-neuron-per-ROI synapse counts, with separate columns
        for ``pre`` (outputs) and ``post`` (inputs).

        .. note::

           In ``roi_counts_df``, the sum of the ``pre`` and ``post`` counts will be more than
           the total ``pre`` and ``post`` values returned in ``neuron_df``.
           That is, synapses are double-counted (or triple-counted, etc.) in ``roi_counts_df``.
           This is because ROIs form a hierarchical structure, so each synapse intersects
           more than one ROI. See :py:func:`.fetch_roi_hierarchy()` for more information.

        .. note::

            Connections which fall outside of all primary ROIs are listed via special entries
            using ``NotPrimary`` in place of an ROI name.  The term ``NotPrimary`` is
            introduced by this function. It isn't used internally by the neuprint database.

    See also:

        :py:func:`.fetch_custom_neurons()` produces similar output,
        but permits you to supply your own cypher query directly.


    Example:

        .. code-block:: ipython

            In [1]: from neuprint import fetch_neurons, NeuronCriteria as NC

            In [2]: neurons_df, roi_counts_df = fetch_neurons(
               ...:     NC(inputRois=['SIP(R)', 'aL(R)'],
               ...:        status='Traced',
               ...:        type='MBON.*'))

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
    criteria.matchvar = 'n'

    # Unlike in fetch_custom_neurons() below, here we specify the
    # return properties individually to avoid a large JSON payload.
    # (Returning a map on every row is ~2x more costly than returning a table of rows/columns.)
    props = compile_columns(client, core_columns=CORE_NEURON_COLS)
    return_exprs = ',\n'.join(f'n.{prop} as {prop}' for prop in props)
    return_exprs = indent(return_exprs, ' '*15)[15:]

    q = f"""\
        {criteria.global_with(prefix=8)}
        MATCH (n :{criteria.label})
        {criteria.all_conditions(prefix=8)}
        RETURN {return_exprs}
        ORDER BY n.bodyId
    """
    neuron_df = client.fetch_custom(q)
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

            .. code-block:: cypher

                ...
                MATCH (n :Neuron)
                ...
                RETURN n
                ...

        client:
            If not provided, the global default ``Client`` will be used.

    Returns:
        Two DataFrames.
        ``(neurons_df, roi_counts_df)``

        In ``neurons_df``, all available columns ``:Neuron`` columns are returned, with the following changes:

            - ROI boolean columns are removed
            - ``roiInfo`` is parsed as json data
            - coordinates (such as ``somaLocation``) are provided as a list ``[x, y, z]``
            - New columns ``inputRoi`` and ``outputRoi`` contain lists of each neuron's ROIs.

        In ``roi_counts_df``, the ``roiInfo`` has been loaded into a table
        of per-neuron-per-ROI synapse counts, with separate columns
        for ``pre`` (outputs) and ``post`` (inputs).

        Connections which fall outside of all primary ROIs are listed via special entries
        using ``NotPrimary`` in place of an ROI name.  The term ``NotPrimary`` is
        introduced by this function. It isn't used internally by the neuprint database.
    """
    results = client.fetch_custom(q)

    if len(results) == 0:
        NEURON_COLS = compile_columns(client, core_columns=CORE_NEURON_COLS)
        neuron_df = pd.DataFrame([], columns=NEURON_COLS, dtype=object)
        roi_counts_df = pd.DataFrame([], columns=['bodyId', 'roi', 'pre', 'post'])
        return neuron_df, roi_counts_df

    neuron_df = pd.DataFrame(results['n'].tolist())

    neuron_df, roi_counts_df = _process_neuron_df(neuron_df, client)
    return neuron_df, roi_counts_df


def _process_neuron_df(neuron_df, client, parse_locs=True):
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
    # Standard columns first, then any extra columns in the results (if any).
    neuron_cols = [*filter(lambda c: c in neuron_df.columns, CORE_NEURON_COLS)]
    extra_cols = {*neuron_df.columns} - {*neuron_cols}
    neuron_cols += [*extra_cols]
    neuron_df = neuron_df[[*neuron_cols]]

    # Make a list of rois for every neuron (both pre and post)
    neuron_df['roiInfo'] = neuron_df['roiInfo'].apply(lambda s: ujson.loads(s))
    neuron_df['inputRois'] = neuron_df['roiInfo'].apply(lambda d: sorted([k for k,v in d.items() if v.get('post')]))
    neuron_df['outputRois'] = neuron_df['roiInfo'].apply(lambda d: sorted([k for k,v in d.items() if v.get('pre')]))

    # Find location columns
    if parse_locs:
        for c in neuron_df.columns:
            if neuron_df[c].dtype != 'object':
                continue
            # Skip columns which contain no dictionaries
            is_dict = [isinstance(x, dict) for x in neuron_df[c]]
            if not any(is_dict):
                continue
            neuron_df.loc[is_dict, c] = neuron_df.loc[is_dict, c].apply(lambda x: x.get('coordinates', x))

    # Return roi info as a separate table.
    # (Note: Some columns aren't present in old neuprint databases.)
    countcols = ['pre', 'post', 'downstream', 'upstream', 'mito']
    countcols = [c for c in countcols if c in neuron_df.columns]
    fullcols = ['bodyId', 'roi', *countcols]
    nonroi_cols = ['bodyId', *countcols]

    roi_counts = [
        {'bodyId': bodyId, 'roi': roi, **counts}
        for bodyId, roiInfo in zip(neuron_df['bodyId'], neuron_df['roiInfo'])
        for roi, counts in roiInfo.items()
    ]
    roi_counts_df = pd.DataFrame(roi_counts, columns=fullcols)
    roi_counts_df = roi_counts_df.fillna(0).astype({c: int for c in countcols})

    # The 'NotPrimary' entries aren't stored by neuprint explicitly.
    # We must compute them by subtracting the summed per-ROI counts
    # from the overall counts in the neuron table.
    roi_totals_df = roi_counts_df.query('roi in @client.primary_rois')[nonroi_cols].groupby('bodyId').sum()
    roi_totals_df = roi_totals_df.reindex(neuron_df['bodyId'])

    not_primary_df = neuron_df[nonroi_cols].set_index('bodyId').fillna(0) - roi_totals_df.fillna(0)
    not_primary_df = not_primary_df.astype(int)
    not_primary_df['roi'] = 'NotPrimary'
    not_primary_df = not_primary_df.reset_index()[fullcols]

    roi_counts_df = pd.concat((roi_counts_df, not_primary_df), ignore_index=True)
    roi_counts_df = roi_counts_df.sort_values(['bodyId', 'roi'], ignore_index=True)

    return neuron_df, roi_counts_df
