import pytest
from unittest.mock import patch, MagicMock
from requests.exceptions import HTTPError
from neuprint import Client
from neuprint.tests import NEUPRINT_SERVER, DATASET


def test_arrow_endpoint_version_check():
    c = Client(NEUPRINT_SERVER, DATASET)
    
    # Test with version higher than 1.7.3
    with patch.object(c, '_fetch_json') as mock_fetch:
        mock_fetch.return_value = {'Version': '1.8.0'}
        assert c.arrow_endpoint() is True
        
    # Test with version exactly 1.7.3
    with patch.object(c, '_fetch_json') as mock_fetch:
        mock_fetch.return_value = {'Version': '1.7.3'}
        assert c.arrow_endpoint() is True
        
    # Test with version lower than 1.7.3
    with patch.object(c, '_fetch_json') as mock_fetch:
        mock_fetch.return_value = {'Version': '1.7.1'}
        assert c.arrow_endpoint() is False
        
    # Test with invalid version format
    with patch.object(c, '_fetch_json') as mock_fetch:
        mock_fetch.return_value = {'Version': 'invalid-version'}
        assert c.arrow_endpoint() is False
        
    # Test with missing Version key
    with patch.object(c, '_fetch_json') as mock_fetch:
        mock_fetch.return_value = {}
        assert c.arrow_endpoint() is False
        
    # Test with exception
    with patch.object(c, '_fetch_json') as mock_fetch:
        mock_fetch.side_effect = Exception("Connection error")
        assert c.arrow_endpoint() is False


def test_fetch_version_error_handling():
    c = Client(NEUPRINT_SERVER, DATASET)
    
    # Test normal operation
    with patch.object(c, '_fetch_json') as mock_fetch:
        mock_fetch.return_value = {'Version': '1.8.0'}
        assert c.fetch_version() == '1.8.0'
    
    # Test HTTP error
    with patch.object(c, '_fetch_json') as mock_fetch:
        mock_fetch.side_effect = HTTPError("404 Client Error: Not Found for url: https://test/api/version")
        with pytest.raises(HTTPError):
            c.fetch_version()
    
    # Test missing Version key
    with patch.object(c, '_fetch_json') as mock_fetch:
        mock_fetch.return_value = {}
        with pytest.raises(KeyError):
            c.fetch_version()
    
    # Test unexpected error
    with patch.object(c, '_fetch_json') as mock_fetch:
        mock_fetch.side_effect = Exception("Unexpected error")
        with pytest.raises(Exception):
            c.fetch_version()


if __name__ == "__main__":
    args = ['-s', '--tb=native', '--pyargs', 'neuprint.tests.test_arrow_endpoint']
    pytest.main(args)