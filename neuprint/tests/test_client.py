import os
import pandas as pd
from neuprint import Client, default_client, set_default_client
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
    
    #assert isinstance(c.fetch_instances(), list)  # Broken. neuprint returns error 500 

    assert isinstance(c.fetch_db_version(), str)
    assert isinstance(c.fetch_profile(), dict)
    assert isinstance(c.fetch_token(), str)
    assert isinstance(c.fetch_daily_type(), tuple)
    assert isinstance(c.fetch_roi_completeness(), pd.DataFrame)
    assert isinstance(c.fetch_roi_connectivity(), pd.DataFrame)
    assert isinstance(c.fetch_skeleton(EXAMPLE_BODY), str)
