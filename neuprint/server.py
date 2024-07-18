''' Server class
    Instantiating this class with a provided server can be used to get a
    list of available datasets.

    Example:

    .. code-block:: ipython

        In [1]: from neuprint import Server

        In [2]: # Create a default server.
           ...: s = Server('neuprint.janelia.org')

        In [3]: s.datasets
        Out[3]:
        ['fib19:v1.0', 'hemibrain:v0.9', 'hemibrain:v1.0.1', 'hemibrain:v1.1', 'hemibrain:v1.2.1', 'manc:v1.0', 'manc:v1.2.1', 'optic-lobe:v1.0']
'''

import os
from requests import Session
import ujson

class Server:
    """
    Server object for interacting with a neuprint server.
    """
    def __init__(self, server, token=None):
        '''
        Args:
            server:
                URL of neuprintHttp server

            token:
                neuPrint token. Either pass explitily as an argument or set
                as ``NEUPRINT_APPLICATION_CREDENTIALS`` environment variable.
                Your token can be retrieved by clicking on your account in
                the NeuPrint web interface.
        '''
        # Token
        if not token:
            token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS')
        if not token:
            raise RuntimeError("No token provided. Please provide one or set NEUPRINT_APPLICATION_CREDENTIALS")
        if ':' in token:
            try:
                token = ujson.loads(token)['token']
            except Exception as ex:
                raise RuntimeError("Did not understand token. Please provide the entire JSON document or (only) the complete token string") from ex
        token = token.replace('"', '')
        # Server
        if '://' not in server:
            server = 'https://' + server
        elif server.startswith('http://'):
            raise RuntimeError("Server must be https, not http")
        elif not server.startswith('https://'):
            protocol = server.split('://')[0]
            raise RuntimeError(f"Unknown protocol: {protocol}")
        self.server = server
        self.token = token
        # Session
        self.session = Session()
        self.session.headers.update({"Authorization": "Bearer " + token,
                                     "Content-type": "application/json"})
        # Datasets
        self.datasets = [*self.fetch_datasets().keys()]


    def _fetch(self, url):
        """
        GET a response from a URL

        Args:
            url: URL
        Returns:
            Response from server
        """
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp

    def _fetch_json(self, url):
        """
        Fetch a response from a URL as JSON

        Args:
            url: URL
        Returns:
            JSON
        """
        resp = self._fetch(url)
        return ujson.loads(resp.content)

    def fetch_datasets(self):
        """
        Fetch the available datasets on the server

        Args:
            None
        Returns:
            dict, keyed by dataset name
        """
        datasets = self._fetch_json(f"{self.server}/api/dbmeta/datasets")
        return datasets
