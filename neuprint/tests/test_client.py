import pytest
import pandas as pd
from neuprint import Client, default_client, set_default_client, clear_default_client, list_all_clients
from neuprint.client import inject_client, _register_client, _user_clients, _thread_copies
from neuprint.tests import NEUPRINT_SERVER, DATASET

EXAMPLE_BODY = 5813037876 # Delta6G, Delta6G_04, Traced, non-cropped


def _make_fake_client(server='s', dataset='d', token='t', verify=True):
    """
    Create a Client without making any network calls.
    We bypass __init__ entirely and just set the attributes
    that __eq__/__hash__/_register_client care about, then register.
    """
    c = Client.__new__(Client)
    c.server = server
    c.dataset = dataset
    c.token = token
    c.verify = verify
    c.session = None
    c.progress = False
    _register_client(c)
    return c


def _reset_client_state():
    clear_default_client()
    _user_clients.clear()
    _thread_copies.clear()


class TestClientRegistration:

    def setup_method(self):
        _reset_client_state()

    def teardown_method(self):
        _reset_client_state()

    def test_two_identical_clients_same_variable(self):
        """
        Regression test: creating two identical clients and assigning
        to the same variable should still leave a usable default.
        """
        c = _make_fake_client('server1', 'ds1')
        c = _make_fake_client('server1', 'ds1')
        dc = default_client()
        assert dc == c

    def test_single_client_is_default(self):
        c = _make_fake_client('server1', 'ds1')
        dc = default_client()
        assert dc == c

    def test_two_different_clients_no_default(self):
        c1 = _make_fake_client('server1', 'ds1')
        c2 = _make_fake_client('server2', 'ds2')
        with pytest.raises(RuntimeError, match="more than one"):
            default_client()

    def test_no_clients_raises(self):
        with pytest.raises(RuntimeError, match="haven't yet created"):
            default_client()

    def test_three_identical_clients(self):
        c1 = _make_fake_client('server1', 'ds1')
        c2 = _make_fake_client('server1', 'ds1')
        c3 = _make_fake_client('server1', 'ds1')
        dc = default_client()
        assert dc == c1  # all equivalent, any is fine

    def test_replace_client_different_dataset(self):
        """Reassigning a variable with a different client should work."""
        c = _make_fake_client('server1', 'ds1')
        c = _make_fake_client('server1', 'ds2')
        # Old client was GC'd, only new one remains
        dc = default_client()
        assert dc.dataset == 'ds2'

    def test_list_all_clients(self):
        c1 = _make_fake_client('server1', 'ds1')
        c2 = _make_fake_client('server2', 'ds2')
        live = list_all_clients()
        assert len(live) == 2
        assert c1 in live
        assert c2 in live

    def test_set_default_client_explicit(self):
        c1 = _make_fake_client('server1', 'ds1')
        c2 = _make_fake_client('server2', 'ds2')
        set_default_client(c2)
        dc = default_client()
        assert dc == c2


def test_members():
    set_default_client(None)
    with pytest.raises(RuntimeError):
        default_client()
    c = Client(NEUPRINT_SERVER, DATASET)
    assert c.server == f'https://{NEUPRINT_SERVER}'
    assert c.dataset == DATASET

    assert default_client() == c

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
    assert isinstance(c.fetch_skeleton(EXAMPLE_BODY), pd.DataFrame)
    assert isinstance(c.fetch_neuron_keys(), list)


def test_fetch_skeleton():
    c = Client(NEUPRINT_SERVER, DATASET)
    orig_df = c.fetch_skeleton(5813027016, False)
    healed_df = c.fetch_skeleton(5813027016, True)

    assert len(orig_df) == len(healed_df)
    assert (healed_df['link'] == -1).sum() == 1
    assert healed_df['link'].iloc[0] == -1


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
    c = Client(NEUPRINT_SERVER, DATASET, verify=True)
    c2 = Client(NEUPRINT_SERVER, DATASET, verify=False)

    set_default_client(c)

    @inject_client
    def f(*, client):
        return client

    # Uses default client unless client was specified
    assert f() == c
    assert f(client=c2) == c2

    with pytest.raises(AssertionError):
        # Wrong signature -- asserts
        @inject_client
        def f2(client):
            pass


if __name__ == "__main__":
    args = ['-s', '--tb=native', '--pyargs', 'neuprint.tests.test_client']
    #args += ['-k', 'fetch_skeleton']
    pytest.main(args)
