import pytest
import pandas as pd

from neuprint import Client, default_client, set_default_client
from neuprint import (fetch_custom, SegmentCriteria as SC, fetch_neurons, fetch_meta,
                      fetch_all_rois, fetch_primary_rois, fetch_simple_connections,
                      fetch_adjacencies, fetch_shortest_paths)
from neuprint.tests import NEUPRINT_SERVER, DATASET

@pytest.fixture(scope='module')
def client():
    c = Client(NEUPRINT_SERVER, DATASET)
    set_default_client(c)
    assert default_client() is c
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
    #neurons, roi_counts = fetch_neurons(SC())

    neurons, roi_counts = fetch_neurons(SC(bodyId=bodyId))
    assert len(neurons) == len(bodyId)
    assert set(roi_counts['bodyId']) == set(bodyId)
    
    neurons, roi_counts = fetch_neurons(SC(instance='APL_R'))
    assert len(neurons) == 1, "There's only one APL neuron in the hemibrain"
    assert neurons.loc[0, 'type'] == "APL"
    assert neurons.loc[0, 'instance'] == "APL_R"

    neurons, roi_counts = fetch_neurons(SC(instance='APL[^ ]*', regex=True))
    assert len(neurons) == 1, "There's only one APL neuron in the hemibrain"
    assert neurons.loc[0, 'type'] == "APL"
    assert neurons.loc[0, 'instance'] == "APL_R"

    neurons, roi_counts = fetch_neurons(SC(type='APL.*', regex=True))
    assert len(neurons) == 1, "There's only one APL neuron in the hemibrain"
    assert neurons.loc[0, 'type'] == "APL"
    assert neurons.loc[0, 'instance'] == "APL_R"

    neurons, roi_counts = fetch_neurons(SC(status=['Traced', 'Orphan'], cropped=False))
    assert neurons.eval('status == "Traced" or status == "Orphan"').all()
    assert not neurons['cropped'].any() 
    
    neurons, roi_counts = fetch_neurons(SC(inputRois='AL(R)', outputRois='SNP(R)'))
    assert all(['AL(R)' in rois for rois in neurons['inputRois']])
    assert all(['SNP(R)' in rois for rois in neurons['outputRois']])
    assert sorted(roi_counts.query('roi == "AL(R)" and post > 0')['bodyId']) == sorted(neurons['bodyId'])
    assert sorted(roi_counts.query('roi == "SNP(R)" and pre > 0')['bodyId']) == sorted(neurons['bodyId'])

    neurons, roi_counts = fetch_neurons(SC(min_pre=1000, min_post=2000))
    assert neurons.eval('pre >= 1000 and post >= 2000').all()


def test_fetch_simple_connections(client):
    bodyId = [294792184, 329566174, 329599710, 417199910, 420274150,
              424379864, 425790257, 451982486, 480927537, 481268653]

    conn_df = fetch_simple_connections(SC(bodyId=bodyId))
    assert set(conn_df['bodyId_pre'].unique()) == set(bodyId)

    conn_df = fetch_simple_connections(None, SC(bodyId=bodyId))
    assert set(conn_df['bodyId_post'].unique()) == set(bodyId)
    
    APL_R = 425790257

    conn_df = fetch_simple_connections(SC(instance='APL_R'))
    assert (conn_df['bodyId_pre'] == APL_R).all()

    conn_df = fetch_simple_connections(SC(type='APL'))
    assert (conn_df['bodyId_pre'] == APL_R).all()

    conn_df = fetch_simple_connections(None, SC(instance='APL_R'))
    assert (conn_df['bodyId_post'] == APL_R).all()

    conn_df = fetch_simple_connections(None, SC(type='APL'))
    assert (conn_df['bodyId_post'] == APL_R).all()

    conn_df = fetch_simple_connections(SC(bodyId=APL_R), min_weight=10)
    assert (conn_df['bodyId_pre'] == APL_R).all()
    assert (conn_df['weight'] >= 10).all()
    
    conn_df = fetch_simple_connections(SC(bodyId=APL_R), min_weight=10, properties=['somaLocation'])
    assert 'somaLocation_pre' in conn_df
    assert 'somaLocation_post' in conn_df

    conn_df = fetch_simple_connections(SC(bodyId=APL_R), min_weight=10, properties=['roiInfo'])
    assert 'roiInfo_pre' in conn_df
    assert 'roiInfo_post' in conn_df
    assert isinstance(conn_df['roiInfo_pre'].iloc[0], dict)


def test_fetch_shortest_paths(client):
    src = 329566174
    dst = 294792184
    paths_df = fetch_shortest_paths(src, dst, min_weight=10)
    assert (paths_df.groupby('path')['bodyId'].first() == src).all()
    assert (paths_df.groupby('path')['bodyId'].last() == dst).all()

    assert (paths_df.groupby('path')['weight'].first() == 0).all()


@pytest.mark.skip
def test_fetch_traced_adjacencies(client):
    pass


def test_fetch_adjacencies(client):
    bodies = [294792184, 329566174, 329599710, 417199910, 420274150,
              424379864, 425790257, 451982486, 480927537, 481268653]
    neuron_df, roi_conn_df = fetch_adjacencies(SC(bodyId=bodies), SC(bodyId=bodies))

    # Should not include non-primary ROIs (except 'NotPrimary')
    assert not ({*roi_conn_df['roi'].unique()} - {*fetch_primary_rois()} - {'NotPrimary'})

    #
    # For backwards compatibility with the previous API,
    # You can also pass a list of bodyIds to this function (instead of SegmentCriteria).
    #
    bodies = [294792184, 329566174, 329599710, 417199910, 420274150,
              424379864, 425790257, 451982486, 480927537, 481268653]
    neuron_df2, roi_conn_df2 = fetch_adjacencies(bodies, bodies)

    # Should not include non-primary ROIs (except 'NotPrimary')
    assert not ({*roi_conn_df2['roi'].unique()} - {*fetch_primary_rois()} - {'NotPrimary'})

    assert (neuron_df.fillna('') == neuron_df2.fillna('')).all().all()
    assert (roi_conn_df == roi_conn_df2).all().all()

def test_fetch_meta(client):
    meta = fetch_meta()
    assert isinstance(meta, dict)


def test_fetch_all_rois(client):
    all_rois = fetch_all_rois()
    assert isinstance(all_rois, list)


def test_fetch_primary_rois(client):
    primary_rois = fetch_primary_rois()
    assert isinstance(primary_rois, list)

if __name__ == "__main__":
    args = ['-s', '--tb=native', '--pyargs', 'neuprint.tests.test_queries']
    #args += ['-k', 'fetch_adjacencies']
    pytest.main(args)
