import pytest
import pandas as pd
from neuprint import Client, default_client, set_default_client, inject_client
from neuprint.tests import NEUPRINT_SERVER, DATASET

EXAMPLE_BODY = 5813037876 # Delta6G, Delta6G_04, Traced, non-cropped


def test_members():
    set_default_client(None)
    assert default_client() is None
    c = Client(NEUPRINT_SERVER, DATASET)
    assert c.server == f'https://{NEUPRINT_SERVER}'
    assert c.dataset == DATASET
    
    assert default_client() is c
    
    df = c.fetch_custom("MATCH (m:Meta) RETURN m.primaryRois as rois")
    assert isinstance(df, pd.DataFrame)
    assert df.columns == ['rois']
    assert len(df) == 1
    assert isinstance(df['rois'].iloc[0], list)
    
    
    assert isinstance(c.fetch_available(), list)
    assert isinstance(c.fetch_help(), str)
    assert c.fetch_server_info() is True
    assert isinstance(c.fetch_version(), str)
    assert isinstance(c.fetch_database(), dict)
    assert isinstance(c.fetch_datasets(), dict)
    assert isinstance(c.fetch_db_version(), str)
    assert isinstance(c.fetch_profile(), dict)
    assert isinstance(c.fetch_token(), str)
    assert isinstance(c.fetch_daily_type(), tuple)
    assert isinstance(c.fetch_roi_completeness(), pd.DataFrame)
    assert isinstance(c.fetch_roi_connectivity(), pd.DataFrame)
    assert isinstance(c.fetch_roi_mesh('AB(R)'), bytes)
    assert isinstance(c.fetch_skeleton(EXAMPLE_BODY), str)


@pytest.mark.xfail
def test_broken_members():
    """
    These endpoints are listed in the neuprintHTTP API,
    but don't seem to work.
    """    
    c = Client(NEUPRINT_SERVER, DATASET)

    # Broken. neuprint returns error 500
    assert isinstance(c.fetch_instances(), list) 


@pytest.mark.skip
def test_keyvalue():
    # TODO:
    # What is an appropriate key/value to test with?
    c = Client(NEUPRINT_SERVER, DATASET)
    c.post_raw_keyvalue(instance, key, b'test-test-test')
    c.fetch_raw_keyvalue(instance, key)


def test_inject_client():
    c = Client(NEUPRINT_SERVER, DATASET)
    c2 = Client(NEUPRINT_SERVER, DATASET)

    set_default_client(c)

    @inject_client
    def f(*, client):
        return client
    
    # Uses default client unless client was specified
    assert f() is c
    assert f(client=c2) is c2

    with pytest.raises(AssertionError):
        # Wrong signature -- asserts
        @inject_client
        def f2(client):
            pass
