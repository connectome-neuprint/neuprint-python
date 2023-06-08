import pytest
import numpy as np
import pandas as pd
import networkx as nx

from neuprint import Client, default_client, set_default_client
from neuprint import (fetch_skeleton, heal_skeleton, reorient_skeleton, skeleton_df_to_nx, skeleton_df_to_swc, skeleton_swc_to_df)

from neuprint.tests import NEUPRINT_SERVER, DATASET


@pytest.fixture(scope='module')
def client():
    c = Client(NEUPRINT_SERVER, DATASET)
    set_default_client(c)
    assert default_client() == c
    return c


@pytest.fixture
def linear_skeleton():
    """
    A test fixture to produce a fake 'skeleton'
    with no branches, just 10 nodes in a line.
    """
    rows = np.arange(1,11)
    coords = np.zeros((10,3), dtype=int)
    coords[:,0] = rows**2
    radii = rows.astype(np.float32)
    links = [-1, *range(1,10)]

    df = pd.DataFrame({'rowId': rows,
                       'x': coords[:,0],
                       'y': coords[:,1],
                       'z': coords[:,2],
                       'radius': radii,
                       'link': links})
    return df


def test_skeleton_df_to_nx(linear_skeleton):
    g = skeleton_df_to_nx(linear_skeleton, directed=False)
    assert not isinstance(g, nx.DiGraph)
    expected_edges = linear_skeleton[['rowId', 'link']].values[1:]
    expected_edges.sort(axis=1)
    assert (np.array(g.edges) == expected_edges).all()

    g = skeleton_df_to_nx(linear_skeleton, directed=True)
    assert isinstance(g, nx.DiGraph)
    assert (np.array(g.edges) == linear_skeleton[['rowId', 'link']].values[1:]).all()

    g = skeleton_df_to_nx(linear_skeleton, with_attributes=True)
    assert (np.array(g.edges) == linear_skeleton[['rowId', 'link']].values[1:]).all()
    for row in linear_skeleton.itertuples():
        attrs = g.nodes[row.rowId]
        assert tuple(attrs[k] for k in [*'xyz', 'radius']) == (row.x, row.y, row.z, row.radius)


def test_skeleton_df_to_swc(linear_skeleton):
    swc = skeleton_df_to_swc(linear_skeleton)
    roundtrip_df = skeleton_swc_to_df(swc)
    assert (roundtrip_df == linear_skeleton).all().all()


def test_reorient_skeleton(linear_skeleton):
    s = linear_skeleton.copy()
    reorient_skeleton(s, 10)
    assert (s['link'] == [*range(2,11), -1]).all()

    s = linear_skeleton.copy()
    reorient_skeleton(s, xyz=(100,0,0))
    assert (s['link'] == [*range(2,11), -1]).all()

    s = linear_skeleton.copy()
    reorient_skeleton(s, use_max_radius=True)
    assert (s['link'] == [*range(2,11), -1]).all()


def test_reorient_broken_skeleton(linear_skeleton):
    broken_skeleton = linear_skeleton.copy()
    broken_skeleton.loc[2, 'link'] = -1
    broken_skeleton.loc[7, 'link'] = -1

    s = broken_skeleton.copy()
    reorient_skeleton(s, 10)
    assert (s['link'].iloc[7:10] == [9,10,-1]).all()

    # reorienting shouldn't change the number of roots,
    # though they may change locations.
    assert len(s.query('link == -1')) == 3


def test_heal_skeleton(linear_skeleton):
    broken_skeleton = linear_skeleton.copy()
    broken_skeleton.loc[2, 'link'] = -1
    broken_skeleton.loc[7, 'link'] = -1

    healed_skeleton = heal_skeleton(broken_skeleton)
    assert (healed_skeleton == linear_skeleton).all().all()


def test_heal_skeleton_with_threshold(linear_skeleton):
    broken_skeleton = linear_skeleton.copy()
    broken_skeleton.loc[2, 'link'] = -1
    broken_skeleton.loc[7, 'link'] = -1

    healed_skeleton = heal_skeleton(broken_skeleton, 10.0)

    # With a threshold of 10, the first break could be healed,
    # but not the second.
    expected_skeleton = linear_skeleton.copy()
    expected_skeleton.loc[7, 'link'] = -1
    assert (healed_skeleton == expected_skeleton).all().all()


def test_fetch_skeleton(client):
    orig_df = fetch_skeleton(5813027016, False)
    healed_df = fetch_skeleton(5813027016, True)

    assert len(orig_df) == len(healed_df)
    assert (healed_df['link'] == -1).sum() == 1
    assert healed_df['link'].iloc[0] == -1


@pytest.mark.skip("Need to write a test for skeleton_segments()")
def test_skeleton_segments(linear_skeleton):
    pass


if __name__ == "__main__":
    args = ['-s', '--tb=native', '--pyargs', 'neuprint.tests.test_skeleton']
    #args += ['-k', 'heal_skeleton']
    pytest.main(args)
