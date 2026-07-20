import warnings

import pytest
import numpy as np
import pandas as pd

from neuprint import Client, default_client, set_default_client
from neuprint import (NeuronCriteria as NC,
                      MitoCriteria as MC,
                      SynapseCriteria as SC,
                      fetch_custom, fetch_neurons, fetch_meta,
                      fetch_all_rois, fetch_primary_rois, fetch_simple_connections,
                      fetch_common_connectivity,
                      fetch_adjacencies, fetch_shortest_paths, fetch_paths,
                      fetch_mitochondria, fetch_synapses_and_closest_mitochondria,
                      fetch_synapses, fetch_mean_synapses, fetch_synapse_connections)

from neuprint.queries.neurons import CORE_NEURON_COLS
from neuprint.tests import NEUPRINT_SERVER, DATASET, TOKEN

@pytest.fixture(scope='module')
def client():
    c = Client(NEUPRINT_SERVER, DATASET)
    set_default_client(c)
    assert default_client() == c
    return c


def test_fetch_custom(client):
    df = fetch_custom("MATCH (m:Meta) RETURN m.primaryRois as rois")
    assert isinstance(df, pd.DataFrame)
    assert df.columns == ['rois']
    assert len(df) == 1
    assert isinstance(df['rois'].iloc[0], list)


def test_fetch_neurons(client):
    bodyId = [294792184, 329566174, 329599710, 417199910, 420274150,
              424379864, 425790257, 451982486, 480927537, 481268653]

    # This works but takes a long time.
    #neurons, roi_counts = fetch_neurons(NC())

    neurons, roi_counts = fetch_neurons(NC(bodyId=bodyId))
    assert len(neurons) == len(bodyId)
    assert set(roi_counts['bodyId']) == set(bodyId)

    neurons, roi_counts = fetch_neurons(NC(instance='APL_R'))
    assert len(neurons) == 1, "There's only one APL neuron in the hemibrain"
    assert neurons.loc[0, 'type'] == "APL"
    assert neurons.loc[0, 'instance'] == "APL_R"

    neurons, roi_counts = fetch_neurons(NC(instance='APL[^ ]*', regex=True))
    assert len(neurons) == 1, "There's only one APL neuron in the hemibrain"
    assert neurons.loc[0, 'type'] == "APL"
    assert neurons.loc[0, 'instance'] == "APL_R"

    neurons, roi_counts = fetch_neurons(NC(type='APL.*', regex=True))
    assert len(neurons) == 1, "There's only one APL neuron in the hemibrain"
    assert neurons.loc[0, 'type'] == "APL"
    assert neurons.loc[0, 'instance'] == "APL_R"

    neurons, roi_counts = fetch_neurons(NC(type=['.*01', '.*02'], regex=True))
    assert len(neurons), "Didn't find any neurons of the given type pattern"
    assert all(lambda t: t.endswith('01') or t.endswith('02') for t in neurons['type'])
    assert any(lambda t: t.endswith('01') for t in neurons['type'])
    assert any(lambda t: t.endswith('02') for t in neurons['type'])

    neurons, roi_counts = fetch_neurons(NC(instance=['.*_L', '.*_R'], regex=True))
    assert len(neurons), "Didn't find any neurons of the given instance pattern"
    assert all(lambda t: t.endswith('_L') or t.endswith('_R') for t in neurons['instance'])

    neurons, roi_counts = fetch_neurons(NC(status=['Traced', 'Orphan'], cropped=False))
    assert neurons.eval('status == "Traced" or status == "Orphan"').all()
    assert not neurons['cropped'].any()

    neurons, roi_counts = fetch_neurons(NC(inputRois='AL(R)', outputRois='SNP(R)'))
    assert all(['AL(R)' in rois for rois in neurons['inputRois']])
    assert all(['SNP(R)' in rois for rois in neurons['outputRois']])
    assert sorted(roi_counts.query('roi == "AL(R)" and post > 0')['bodyId']) == sorted(neurons['bodyId'])
    assert sorted(roi_counts.query('roi == "SNP(R)" and pre > 0')['bodyId']) == sorted(neurons['bodyId'])

    neurons, roi_counts = fetch_neurons(NC(min_pre=1000, min_post=2000))
    assert neurons.eval('pre >= 1000 and post >= 2000').all()

    neurons, roi_counts = fetch_neurons(NC(bodyId=bodyId), returned_columns="core")
    # hemibrain dataset has all the CORE_NEURON_COLS
    assert set(neurons.columns) == set(CORE_NEURON_COLS)

    requested_columns = ['bodyId', 'instance']
    neurons = fetch_neurons(NC(bodyId=bodyId), returned_columns=requested_columns, omit_rois=True)
    assert set(neurons.columns) == set(requested_columns)



def test_fetch_simple_connections(client):
    bodyId = [294792184, 329566174, 329599710, 417199910, 420274150,
              424379864, 425790257, 451982486, 480927537, 481268653]

    conn_df = fetch_simple_connections(NC(bodyId=bodyId))
    assert set(conn_df['bodyId_pre'].unique()) == set(bodyId)

    conn_df = fetch_simple_connections(None, NC(bodyId=bodyId))
    assert set(conn_df['bodyId_post'].unique()) == set(bodyId)

    APL_R = 425790257

    conn_df = fetch_simple_connections(NC(instance='APL_R'))
    assert (conn_df['bodyId_pre'] == APL_R).all()

    conn_df = fetch_simple_connections(NC(type='APL'))
    assert (conn_df['bodyId_pre'] == APL_R).all()

    conn_df = fetch_simple_connections(None, NC(instance='APL_R'))
    assert (conn_df['bodyId_post'] == APL_R).all()

    conn_df = fetch_simple_connections(None, NC(type='APL'))
    assert (conn_df['bodyId_post'] == APL_R).all()

    conn_df = fetch_simple_connections(NC(bodyId=APL_R), min_weight=10)
    assert (conn_df['bodyId_pre'] == APL_R).all()
    assert (conn_df['weight'] >= 10).all()

    conn_df = fetch_simple_connections(NC(bodyId=APL_R), min_weight=10, properties=['somaLocation'])
    assert 'somaLocation_pre' in conn_df
    assert 'somaLocation_post' in conn_df

    conn_df = fetch_simple_connections(NC(bodyId=APL_R), min_weight=10, properties=['roiInfo'])
    assert 'roiInfo_pre' in conn_df
    assert 'roiInfo_post' in conn_df
    assert isinstance(conn_df['roiInfo_pre'].iloc[0], dict)


def test_fetch_simple_connections_weight_props(client):
    bodyId = [294792184, 329566174, 329599710, 417199910, 420274150,
              424379864, 425790257, 451982486, 480927537, 481268653]

    # weight_props=['weight'] restores the historical (pre-weight_props) behavior.
    conn_df = fetch_simple_connections(NC(bodyId=bodyId), properties=[], weight_props=['weight'])
    assert conn_df.columns.tolist() == ['bodyId_pre', 'bodyId_post', 'weight', 'conn_roiInfo']

    # weightHP is always available at the edge level.
    conn_df = fetch_simple_connections(NC(bodyId=bodyId), properties=[], weight_props=['weightHP'])
    assert conn_df.columns.tolist() == ['bodyId_pre', 'bodyId_post', 'weight', 'weightHP', 'conn_roiInfo']
    assert pd.api.types.is_integer_dtype(conn_df['weightHP'])
    assert (conn_df['weightHP'] >= 0).all()
    assert (conn_df['weightHP'] <= conn_df['weight']).all()

    # A single property name (not wrapped in a list) is also accepted.
    conn_df_single = fetch_simple_connections(NC(bodyId=bodyId), properties=[], weight_props='weightHP')
    assert (conn_df_single == conn_df).all().all()

    # The default is 'all'.  This test dataset has no axon/dendrite polarity info,
    # so the (unavailable) polarity-split properties are silently omitted, whether
    # via the implicit default or the explicit 'all' shorthand -- no warning expected.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        conn_df = fetch_simple_connections(NC(bodyId=bodyId), properties=[])
        conn_df2 = fetch_simple_connections(NC(bodyId=bodyId), properties=[], weight_props='all')
    assert conn_df.columns.tolist() == ['bodyId_pre', 'bodyId_post', 'weight', 'weightHP', 'conn_roiInfo']
    assert conn_df2.columns.tolist() == conn_df.columns.tolist()

    # But explicitly requesting one of those unavailable properties warns,
    # and the column is omitted entirely (never returned as all-zeros).
    with pytest.warns(UserWarning, match="axon/dendrite polarity"):
        conn_df = fetch_simple_connections(NC(bodyId=bodyId), properties=[],
                                           weight_props=['weightHP', 'weightAxonDendrite'])
    assert conn_df.columns.tolist() == ['bodyId_pre', 'bodyId_post', 'weight', 'weightHP', 'conn_roiInfo']

    # Unrecognized weight_props are rejected.
    with pytest.raises(AssertionError):
        fetch_simple_connections(NC(bodyId=bodyId), weight_props=['bogus'])


def test_fetch_common_connectivity_weight_props(client):
    bodyId = [294792184, 329566174, 329599710, 417199910, 420274150,
              424379864, 425790257, 451982486, 480927537, 481268653]

    # weight_props is simply forwarded to fetch_simple_connections().
    conn_df = fetch_common_connectivity(NC(bodyId=bodyId), weight_props=['weightHP'])
    assert 'weightHP' in conn_df.columns
    assert (conn_df['weightHP'] <= conn_df['weight']).all()


def test_fetch_shortest_paths(client):
    src = 329566174
    dst = 294792184
    paths_df = fetch_shortest_paths(src, dst, min_weight=10)
    assert (paths_df.groupby('path')['bodyId'].first() == src).all()
    assert (paths_df.groupby('path')['bodyId'].last() == dst).all()

    assert (paths_df.groupby('path')['weight'].first() == 0).all()

def test_fetch_paths_exact(client):
    src = 329566174
    dst = 294792184
    paths_df = fetch_paths(src, dst, path_length=2, min_weight=10, timeout=3)
    assert (paths_df.groupby('path')['bodyId'].first() == src).all()
    assert (paths_df.groupby('path')['bodyId'].last() == dst).all()

    assert "path_length" in paths_df.columns
    assert (paths_df['path_length'] == 2).all()

def test_fetch_paths_limited(client):
    src = 329566174
    dst = 294792184
    paths_df = fetch_paths(src, dst, max_path_length=2, min_weight=10, timeout=3)
    assert (paths_df.groupby('path')['bodyId'].first() == src).all()
    assert (paths_df.groupby('path')['bodyId'].last() == dst).all()

    assert "path_length" in paths_df.columns
    assert (paths_df['path_length'] <= 2).all()

def test_fetch_paths_input(client):
    src = 329566174
    dst = 294792184
    with pytest.raises(ValueError):
        # path_length and max_path_length are mutually exclusive
        fetch_paths(src, dst, path_length=2, max_path_length=2, min_weight=10, timeout=3)


@pytest.mark.skip
def test_fetch_traced_adjacencies(client):
    pass


def test_fetch_adjacencies(client):
    bodies = [294792184, 329566174, 329599710, 417199910, 420274150,
              424379864, 425790257, 451982486, 480927537, 481268653]
    neuron_df, roi_conn_df = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies))

    # Should not include non-primary ROIs (except 'NotPrimary')
    assert not ({*roi_conn_df['roi'].unique()} - {*fetch_primary_rois()} - {'NotPrimary'})

    #
    # For backwards compatibility with the previous API,
    # You can also pass a list of bodyIds to this function (instead of NeuronCriteria).
    #
    bodies = [294792184, 329566174, 329599710, 417199910, 420274150,
              424379864, 425790257, 451982486, 480927537, 481268653]
    neuron_df2, roi_conn_df2 = fetch_adjacencies(bodies, bodies)

    # Should not include non-primary ROIs (except 'NotPrimary')
    assert not ({*roi_conn_df2['roi'].unique()} - {*fetch_primary_rois()} - {'NotPrimary'})

    assert (neuron_df.fillna('') == neuron_df2.fillna('')).all().all()
    assert (roi_conn_df == roi_conn_df2).all().all()

    # What happens if results are empty
    neuron_df, roi_conn_df = fetch_adjacencies(879442155, 5813027103)
    assert len(neuron_df) == 0
    assert len(roi_conn_df) == 0
    assert neuron_df.columns.tolist() == ['bodyId', 'instance', 'type']


def test_fetch_adjacencies_omit_rois(client):
    bodies = [294792184, 329566174, 329599710, 417199910, 420274150,
              424379864, 425790257, 451982486, 480927537, 481268653]

    # weight_props=['weight'] here, to keep this test focused on omit_rois
    # semantics rather than coupling it to the (separately tested) weight_props
    # default of 'all'.
    neuron_df, roi_conn_df = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies), weight_props=['weight'])
    neuron_df2, conn_df = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies), weight_props=['weight'], omit_rois=True)

    # The per-ROI breakdown is replaced by one row per body pair.
    assert conn_df.columns.tolist() == ['bodyId_pre', 'bodyId_post', 'weight']

    # The totals must agree with the per-ROI table, aggregated across ROIs.
    expected = (roi_conn_df
                    .groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight']
                    .sum()
                    .sort_values(['bodyId_pre', 'bodyId_post'], ignore_index=True))
    assert (conn_df == expected).all().all()

    # The neuron table is unaffected.
    assert (neuron_df.fillna('') == neuron_df2.fillna('')).all().all()

    # min_total_weight is applied by the server.
    _neuron_df, conn_df = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies), weight_props=['weight'],
                                            min_total_weight=10, omit_rois=True)
    assert (conn_df['weight'] >= 10).all()
    assert (conn_df == expected.query('weight >= 10').reset_index(drop=True)).all().all()

    # Options that depend on per-ROI weights can't be honored.
    with pytest.raises(ValueError):
        fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies), rois=['SLP(R)'], omit_rois=True)
    with pytest.raises(ValueError):
        fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies), min_roi_weight=5, omit_rois=True)
    with pytest.raises(ValueError):
        fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies), include_nonprimary=True, omit_rois=True)

    # What happens if results are empty
    neuron_df, conn_df = fetch_adjacencies(879442155, 5813027103, weight_props=['weight'], omit_rois=True)
    assert len(neuron_df) == 0
    assert len(conn_df) == 0
    assert conn_df.columns.tolist() == ['bodyId_pre', 'bodyId_post', 'weight']


def test_fetch_adjacencies_threads(client):
    bodies = [294792184, 329566174, 329599710, 417199910, 420274150,
              424379864, 425790257, 451982486, 480927537, 481268653]

    # Use a small batch_size so that several batches are actually fetched in parallel.
    kwargs = dict(batch_size=3)

    serial_neurons, serial_conns = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies),
                                                     threads=1, **kwargs)
    threaded_neurons, threaded_conns = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies),
                                                         threads=4, **kwargs)

    # Batches are concatenated in input order, so threading must not perturb the results.
    assert (serial_neurons.fillna('') == threaded_neurons.fillna('')).all().all()
    assert (serial_conns == threaded_conns).all().all()

    # ...and the same must hold for the omit_rois path.
    _n, serial_conns = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies),
                                         omit_rois=True, threads=1, **kwargs)
    _n, threaded_conns = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies),
                                           omit_rois=True, threads=4, **kwargs)
    assert (serial_conns == threaded_conns).all().all()


def test_fetch_adjacencies_threaded_backfill(client):
    # When sources and targets differ, the neurons_df for the bodies on the
    # un-prefetched side is 'backfilled' in batches of 10_000, which are also
    # fetched concurrently.  Use enough downstream Segments to span several batches.
    sources = NC(bodyId=[329566174, 425790257])
    targets = NC(label='Segment')

    serial_neurons, serial_conns = fetch_adjacencies(sources, targets, omit_rois=True, threads=1)
    threaded_neurons, threaded_conns = fetch_adjacencies(sources, targets, omit_rois=True, threads=4)

    # More than one backfill batch, otherwise this test proves nothing.
    assert len(serial_neurons) > 10_000

    assert (serial_neurons.fillna('') == threaded_neurons.fillna('')).all().all()
    assert (serial_conns == threaded_conns).all().all()


def test_fetch_adjacencies_weight_props(client):
    bodies = [294792184, 329566174, 329599710, 417199910, 420274150,
              424379864, 425790257, 451982486, 480927537, 481268653]

    # Even with the new 'all' default, this dataset's roiInfo doesn't break weightHP
    # down per-ROI, so it's silently dropped from the per-ROI table -- the result
    # is identical to the pre-weight_props behavior.
    _n, roi_conn_df = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies))
    assert roi_conn_df.columns.tolist() == ['bodyId_pre', 'bodyId_post', 'roi', 'weight']

    # weightHP is always available at the edge level, but this dataset's roiInfo
    # doesn't break it down per-ROI, so it's dropped (with a warning) from the
    # per-ROI table -- never shown as a meaningless all-zero column.
    with pytest.warns(UserWarning, match="per-ROI breakdown"):
        _n, roi_conn_df = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies), weight_props=['weightHP'])
    assert roi_conn_df.columns.tolist() == ['bodyId_pre', 'bodyId_post', 'roi', 'weight']

    # The flat (omit_rois) table reads weightHP directly off the edge, so it's unaffected.
    _n, conn_df = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies),
                                    weight_props=['weightHP'], omit_rois=True)
    assert conn_df.columns.tolist() == ['bodyId_pre', 'bodyId_post', 'weight', 'weightHP']
    assert pd.api.types.is_integer_dtype(conn_df['weightHP'])
    assert (conn_df['weightHP'] >= 0).all()
    assert (conn_df['weightHP'] <= conn_df['weight']).all()

    # A single property name (not wrapped in a list) is also accepted.
    _n, conn_df_single = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies),
                                           weight_props='weightHP', omit_rois=True)
    assert (conn_df_single == conn_df).all().all()

    # 'all' is a shorthand that silently omits whatever isn't usefully available --
    # both the polarity-split props this dataset lacks entirely, and the roiInfo
    # per-ROI breakdown that weightHP lacks here -- no warnings expected.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _n, roi_conn_df = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies), weight_props='all')
        _n, conn_df = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies), weight_props='all', omit_rois=True)
    assert roi_conn_df.columns.tolist() == ['bodyId_pre', 'bodyId_post', 'roi', 'weight']
    assert conn_df.columns.tolist() == ['bodyId_pre', 'bodyId_post', 'weight', 'weightHP']

    # Explicitly requesting an unavailable polarity-split property warns, and the
    # column is omitted entirely (never returned as all-zeros).
    with pytest.warns(UserWarning, match="axon/dendrite polarity"):
        _n, conn_df = fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies),
                                        weight_props=['weightHP', 'weightAxonDendrite'], omit_rois=True)
    assert conn_df.columns.tolist() == ['bodyId_pre', 'bodyId_post', 'weight', 'weightHP']

    # Unrecognized weight_props are rejected.
    with pytest.raises(AssertionError):
        fetch_adjacencies(NC(bodyId=bodies), NC(bodyId=bodies), weight_props=['bogus'])


def test_fetch_adjacencies_weight_props_polarity_available(client, monkeypatch):
    # Simulate a dataset that DOES have axon/dendrite polarity info (none of the
    # currently available public datasets do), so weightAxonDendrite passes the
    # availability gate.  (This live dataset's roiInfo still won't actually carry a
    # per-ROI weightAxonDendrite breakdown -- the property is only simulated here via
    # 'axonOut' -- so the per-ROI table still omits it, but the flat table includes it.)
    monkeypatch.setattr(client, 'fetch_neuron_keys', lambda: ['bodyId', 'axonOut'])

    bodies = [294792184, 329566174, 329599710, 417199910, 420274150]
    # Pass client= explicitly: fetch_adjacencies() otherwise resolves the default
    # client via default_client(), which hands back a per-thread deepcopy rather
    # than this exact (monkeypatched) instance.
    with pytest.warns(UserWarning, match="per-ROI breakdown"):
        _n, roi_conn_df = fetch_adjacencies(NC(bodyId=bodies, client=client), NC(bodyId=bodies, client=client),
                                            weight_props=['weightAxonDendrite'], client=client)
    assert 'weightAxonDendrite' not in roi_conn_df.columns

    _n, conn_df = fetch_adjacencies(NC(bodyId=bodies, client=client), NC(bodyId=bodies, client=client),
                                    weight_props=['weightAxonDendrite'], omit_rois=True, client=client)
    assert 'weightAxonDendrite' in conn_df.columns
    assert (conn_df['weightAxonDendrite'] == 0).all()


def test_fetch_meta(client):
    meta = fetch_meta()
    assert isinstance(meta, dict)


def test_fetch_all_rois(client):
    all_rois = fetch_all_rois()
    assert isinstance(all_rois, list)


def test_fetch_primary_rois(client):
    primary_rois = fetch_primary_rois()
    assert isinstance(primary_rois, list)


def test_fetch_mitochondria(client):
    nc = NC(type='ExR.*', regex=True, rois=['EB'])
    mc = MC(rois=['FB', 'LAL(R)'], mitoType='dark', size=100_000, primary_only=True)
    mito_df = fetch_mitochondria(nc, mc)
    assert set(mito_df['roi']) == {'FB', 'LAL(R)'}
    assert (mito_df['mitoType'] == 'dark').all()
    assert (mito_df['size'] >= 100_000).all()

    neuron_df, _count_df = fetch_neurons(nc)
    mito_df = mito_df.merge(neuron_df[['bodyId', 'type']], 'left', on='bodyId', suffixes=['_mito', '_body'])
    assert mito_df['type'].isnull().sum() == 0
    assert mito_df['type'].apply(lambda s: s.startswith('ExR')).all()


def test_fetch_synapses(client):
    nc = NC(type='ExR.*', regex=True, rois=['EB'])
    sc = SC(rois=['FB', 'LAL(R)'], primary_only=True)
    syn_df = fetch_synapses(nc, sc)
    assert set(syn_df['roi']) == {'FB', 'LAL(R)'}

    # Ensure proper body set used.
    neuron_df, _count_df = fetch_neurons(nc)
    syn_df = syn_df.merge(neuron_df[['bodyId', 'type']], 'left', on='bodyId', suffixes=['_syn', '_body'])
    assert syn_df['type_body'].isnull().sum() == 0
    assert syn_df['type_body'].apply(lambda s: s.startswith('ExR')).all()


def test_fetch_mean_synapses(client):
    nc = NC(type='ExR.*', regex=True, rois=['EB'])
    sc = SC(rois=['FB', 'LAL(R)'], primary_only=True)
    mean_df = fetch_mean_synapses(nc, sc)
    mean_df = mean_df.sort_values(['bodyId', 'roi', 'type'], ignore_index=True)
    assert set(mean_df['roi']) == {'FB', 'LAL(R)'}

    # Ensure proper body set used.
    neuron_df, _count_df = fetch_neurons(nc)
    mean_df = mean_df.merge(neuron_df[['bodyId', 'type']], 'left', on='bodyId', suffixes=['_syn', '_body'])
    assert mean_df['type_body'].isnull().sum() == 0
    assert mean_df['type_body'].apply(lambda s: s.startswith('ExR')).all()

    # Compare with locally averaged results
    syn_df = fetch_synapses(nc, sc)
    expected_df = syn_df.groupby(['bodyId', 'roi', 'type'], observed=True).agg({'x': ['count', 'mean'], 'y': 'mean', 'z': 'mean', 'confidence': 'mean'}).reset_index()
    expected_df.columns = ['bodyId', 'roi', 'type', 'count', *'xyz', 'confidence']
    expected_df = expected_df.sort_values(['bodyId', 'roi', 'type'], ignore_index=True)
    assert np.allclose(mean_df[[*'xyz', 'confidence']].values, expected_df[[*'xyz', 'confidence']].values)


def test_fetch_synapses_and_closest_mitochondria(client):
    syn_mito_distances = fetch_synapses_and_closest_mitochondria(NC(type='ExR2'), SC(type='pre'))
    assert len(syn_mito_distances), "Shouldn't be empty!"


def test_fetch_synapse_connections(client):
    rois = ['PED(R)', 'SMP(R)']
    syn_df = fetch_synapse_connections(792368888, None, SC(rois=rois, primary_only=True), batch_size=2)
    assert syn_df.eval('roi_pre in @rois and roi_post in @rois').all()
    dtypes = syn_df.dtypes.to_dict()

    # Empty results
    syn_df = fetch_synapse_connections(879442155, 5813027103)
    assert len(syn_df) == 0
    assert syn_df.dtypes.to_dict() == dtypes


def test_fetch_synapses_no_duplicate_columns(client):
    # Regression test: some datasets (e.g. manc) store 'bodyId' and 'roi' properties
    # directly on :Synapse nodes (in addition to the ROI-name boolean flags), which
    # used to leak into the "additional properties" columns and duplicate the
    # already-present 'bodyId'/'roi' columns.
    # Note: this creates a second live Client (for a dataset other than the module-scoped
    # 'client' fixture's), and always passes client= explicitly so it's never relied on
    # as the ambient default. But merely constructing it still perturbs the global default
    # (see _register_client()/set_default_client()), so restore the fixture's client as the
    # default afterward -- otherwise later tests that rely on the implicit default break.
    manc_client = Client(NEUPRINT_SERVER, 'manc:v1.2.3', token=TOKEN)
    try:
        bodies = [10000, 10002]

        syn_df = fetch_synapses(NC(bodyId=bodies, client=manc_client), client=manc_client)
        assert syn_df.columns.tolist().count('bodyId') == 1
        assert syn_df.columns.tolist().count('roi') == 1

        conn_df = fetch_synapse_connections(NC(bodyId=bodies, client=manc_client), client=manc_client)
        assert conn_df.columns.tolist().count('bodyId_pre') == 1
        assert conn_df.columns.tolist().count('bodyId_post') == 1
        assert conn_df.columns.tolist().count('roi_pre') == 1
        assert conn_df.columns.tolist().count('roi_post') == 1
    finally:
        set_default_client(client)


def test_issue_69(client):
    # Issue #69: somaLocation should be a list independent of omit_rois parameter.

    # This body in hemibrain is known to have a somaLocation.
    bodyId = 294792184
    neuron_df, roi_counts_df = fetch_neurons(NC(bodyId=bodyId))
    assert 'somaLocation' in neuron_df.columns
    somaLocation = neuron_df.iloc[0]["somaLocation"]
    assert isinstance(somaLocation, list)
    assert len(somaLocation) == 3

    # Fetch without ROIs
    neuron_df = fetch_neurons(NC(bodyId=bodyId), omit_rois=True)
    assert 'somaLocation' in neuron_df.columns
    somaLocation = neuron_df.iloc[0]["somaLocation"]
    assert isinstance(somaLocation, list)
    assert len(somaLocation) == 3


if __name__ == "__main__":
    args = ['-s', '--tb=native', '--pyargs', 'neuprint.tests.test_queries']
    #args += ['-k', 'test_fetch_synapse_connections']
    #args += ['-k', 'fetch_synapses_and_closest_mitochondria']
    #args += ['-k', 'fetch_mean_synapses']
    pytest.main(args)
