'''
The ``client`` module contains the ``Client`` object and related utility functions.

All communication to the neuPrint server is peformed using a ``Client`` object.
Holds your authorization credentials, the dataset name to use,
and other connection settings.

Most ``neuprint-python`` functions do not require you to explicitly
provide a Client object to use. Instead, the first ``Client`` you
create will be stored as the default ``Client`` to be used with all
``neuprint-python`` functions if you don't explicitly specify one.

Example:

    .. code-block:: ipython
    
        In [1]: from neuprint import Client, fetch_custom, fetch_neurons

        In [2]: c = Client('neuprint.janelia.org', dataset='hemibrain:v1.0')

        In [3]: fetch_custom("""\\
           ...:     MATCH (n: Neuron)
           ...:     WHERE n.status = "Traced" AND NOT n.cropped
           ...:     RETURN n.bodyId as bodyId, n.type as type, n.instance as instance
           ...:     ORDER BY n.type, n.instance
           ...: """)
        Out[3]:
                   bodyId        type             instance
        0       511051477   5th s-LNv            5th s-LNv
        1       947590512  ADL01a_pct  ADL01a_pct(ADL01)_R
        2      1100952886  ADL01b_pct  ADL01b_pct(ADL01)_R
        3      1228484534  ADL01b_pct  ADL01b_pct(ADL01)_R
        4      1290563000  ADL01b_pct  ADL01b_pct(ADL01)_R
        ...           ...         ...                  ...
        21658  2346523421        None                 None
        21659  2397377415        None                 None
        21660  2429314661        None                 None
        21661  2464541644        None                 None
        21662  2404203061        None                 None
        
        [21663 rows x 3 columns]

Tip:
    
    All cypher queries are logged, but the messages are not
    shown in the console by default.  To display them, see
    :py:func:`setup_debug_logging()`.
'''
import os
import sys
import copy
import inspect
import logging
import functools
import threading
import collections
from textwrap import dedent, indent

import pandas as pd

import urllib3
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning

from requests import Session, RequestException, HTTPError
from requests.adapters import HTTPAdapter

# ujson is faster than Python's builtin json module
import ujson

logger = logging.getLogger(__name__)
DEFAULT_NEUPRINT_CLIENT = None
NEUPRINT_CLIENTS = {}


def setup_debug_logging():
    """
    Simple debug logging configuration.
    Useful for interactive terminal sessions.
    
    Warning:
        Replaces your current logging setup.
        If you've already set up logging for your app,
        don't call this function.
        Enable neuprint debug logging via:
        
        .. code-block:: python
        
            import logging
            logging.getLogger('neuprint.client').setLevel(logging.DEBUG)

    To disable cypher logging again, increase the logging severity threshold:
    
        .. code-block:: python
        
            import logging
            logging.getLogger('neuprint.client').setLevel(logging.INFO)

    Example:
    
        .. code-block:: ipython
        
            In [1]: from neuprint.client import setup_debug_logging
               ...: from neuprint import fetch_neurons
               ...:
               ...: setup_debug_logging()
               ...: neuron_df, roi_df = fetch_neurons(SC(type='MBON.*', rois=['MB(R)'], regex=True))
            [2020-01-30 08:48:20,367] DEBUG Performing cypher query against dataset 'hemibrain:v1.0':
                MATCH (n :Neuron)
                // -- Basic conditions for segment 'n' --
                WHERE
                  n.type =~ 'MBON.*'
                  AND (n.`MB(R)`)
                RETURN n
                ORDER BY n.bodyId
    """
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    logger.setLevel(logging.DEBUG)


def default_client():
    """
    Obtain the default Client object to use.
    This function returns a separate copy of the
    default client for each thread (and process).
    
    There's usually no need to call this function.
    It is automatically called by all query functions if
    you haven't passed in an explict `client` argument.
    """
    global DEFAULT_NEUPRINT_CLIENT

    thread_id = threading.current_thread().ident
    pid = os.getpid()

    try:
        c = NEUPRINT_CLIENTS[(thread_id, pid)]
    except KeyError:
        if DEFAULT_NEUPRINT_CLIENT is None:
            raise RuntimeError(
                    "No default Client has been set yet. "
                    "Please create a Client object to serve as the default")

        c = copy.deepcopy(DEFAULT_NEUPRINT_CLIENT)
        NEUPRINT_CLIENTS[(thread_id, pid)] = c

    return c


def set_default_client(client):
    """
    Set (or overwrite) the default Client.
    
    There's usually no need to call this function.
    It's is automatically called when your first
    ``Client`` is created, but you can call it again
    to replace the default.
    """
    global NEUPRINT_CLIENTS
    global DEFAULT_NEUPRINT_CLIENT

    thread_id = threading.current_thread().ident
    pid = os.getpid()

    DEFAULT_NEUPRINT_CLIENT = client
    NEUPRINT_CLIENTS.clear()
    NEUPRINT_CLIENTS[(thread_id, pid)] = client


def inject_client(f):
    """
    Decorator.
    Injects the default 'client' as a keyword argument
    onto the decorated function, if the user hasn't supplied
    one herself.

    In typical usage the user will create one Client object,
    and use it with every neuprint function.
    Rather than requiring the user to pass the the client
    to every neuprint call, this decorator automatically
    passes the default (global) Client.
    """
    argspec = inspect.getfullargspec(f)
    assert 'client' in argspec.kwonlyargs, \
        f"Cannot wrap {f.__name__}: neuprint API wrappers must accept 'client' as a keyword-only argument."
    
    @functools.wraps(f)
    def wrapper(*args, client=None, **kwargs):
        if client is None:
            client = default_client()
        return f(*args, **kwargs, client=client)

    wrapper.__signature__ = inspect.signature(f)
    return wrapper


def verbose_errors(f):
    """
    Decorator to be used with functions that directly fetch from neuprint.
    If the decorated function fails due to a RequestException,
    extra information is added to the exception text.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except RequestException as ex:
            # If the error response body is non-empty, show that in the traceback, too.
            # neuprint-http error messages are often helpful -- show them!
            if hasattr(ex, 'response_content_appended') or (ex.response is None and ex.request is None):
                raise # Nothing to add to the exception.

            msg = ""

            # Show the endpoint
            if (ex.request is not None):
                msg += f"Error accessing {ex.request.method} {ex.request.url}\n"

            if isinstance(ex, HTTPError):
                # If the user's arguments included a 'json' argument
                # containing a 'cypher' key query, show it.
                callargs = inspect.getcallargs(f, *args, **kwargs)
                if 'json' in callargs and isinstance(callargs['json'], collections.abc.Mapping):
                    cypher = callargs['json'].get('cypher')
                    if cypher:
                        msg += f"\nCypher was:\n\n{cypher}\n"

            # Show the server's error message
            if ex.response is not None:
                msg += f"\nReturned Error"
                if hasattr(ex.response, 'status_code'):
                    msg += f" ({ex.response.status_code})"
                if ex.response.content:
                    try:
                        err = ex.response.json()['error']
                        msg += f":\n\n{err}"
                    except Exception:
                        pass
            
            new_ex = copy.copy(ex)
            new_ex.args = (msg, *ex.args[1:])
            
            # In case this decorator is used twice in a nested call,
            # mark it as already modified it doesn't get modified twice.
            new_ex.response_content_appended = True
            raise new_ex from ex
    return wrapper


class Client:
    def __init__(self, server, dataset=None, token=None, verify=True):
        """
        Client constructor.
        
        Args:
            server:
                URL of neuprintHttp server

            token:
                neuPrint token. Either pass explitily as an argument or set
                as ``NEUPRINT_APPLICATION_CREDENTIALS`` environment variable.
                Your token can be retrieved by clicking on your account in
                the NeuPrint web interface.

            verify:
                If ``True`` (default), enforce signed credentials.

            dataset:
                The dataset to run all queries against, e.g. 'hemibrain'.
                If not provided, the server will use a default dataset for
                all queries.
        """
        if not token:
            token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS')

        if not token:
            raise RuntimeError("No token provided. Please provide one or set NEUPRINT_APPLICATION_CREDENTIALS")

        if ':' in token:
            try:
                token = ujson.loads(token)['token']
            except Exception:
                raise RuntimeError("Did not understand token. Please provide the entire JSON document or (only) the complete token string")

        token = token.replace('"', '')

        if '://' not in server:
            server = 'https://' + server
        elif server.startswith('http://'):
            raise RuntimeError("Server must be https, not http")
        elif not server.startswith('https://'):
            protocol = server.split('://')[0]
            raise RuntimeError(f"Unknown protocol: {protocol}")

        # Remove trailing backslash
        while server.endswith('/'):
            server = server[:-1]

        self.server = server

        self.session = Session()
        self.session.headers.update({"Authorization": "Bearer " + token,
                                     "Content-type": "application/json"})

        # If the connection fails, retry a couple times.
        retries = Retry(connect=2, backoff_factor=0.1)
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

        self.verify = verify
        if not verify:
            urllib3.disable_warnings(InsecureRequestWarning)

        all_datasets = [*self.fetch_datasets().keys()]
        if len(all_datasets) == 0:
            raise RuntimeError(f"The neuprint server {self.server} has no datasets!")

        if len(all_datasets) == 1 and not dataset:
            self.dataset = all_datasets[0]
            logger.info(f"Initializing neuprint.Client with dataset: {self.dataset}")
        elif dataset in all_datasets:
            self.dataset = dataset
        else:
            raise RuntimeError(f"Dataset '{dataset}' does not exist on"
                               f" the neuprint server ({self.server}).\n"
                               f"Available datasets: {all_datasets}")

        # Set this as the default client if there isn't one already
        global DEFAULT_NEUPRINT_CLIENT
        if DEFAULT_NEUPRINT_CLIENT is None:
            set_default_client(self)

        from .queries import fetch_meta, _all_rois_from_meta
        # Pre-cache these metadata fields,
        # to avoid re-fetching them for many queries that need them.
        self.meta = fetch_meta(client=self)
        self.primary_rois = self.meta['primaryRois']
        self.all_rois = _all_rois_from_meta(self.meta)


    def __repr__(self):
        s = f'Client("{self.server}", "{self.dataset}"'
        if not self.verify:
            s += ", verify=False"
        s += ")"
        return s

    @verbose_errors
    def _fetch(self, url, json=None, ispost=False):
        if ispost:
            r = self.session.post(url, json=json, verify=self.verify)
        else:
            assert json is None, "Can't provide a body via GET method"
            r = self.session.get(url, verify=self.verify)
        r.raise_for_status()
        return r


    def _fetch_raw(self, url, json=None, ispost=False):
        return self._fetch(url, json=json, ispost=ispost).content


    def _fetch_json(self, url, json=None, ispost=False):
        r = self._fetch(url, json=json, ispost=ispost)
        return ujson.loads(r.content)


    ##
    ## CUSTOM QUERIES
    ##
    ## Note: Transaction queries are not implemented here.  See admin.py

    def fetch_custom(self, cypher, dataset="", format='pandas'):
        """
        Query the neuprint server with a custom Cypher query.
        
        Args:
            cypher:
                A cypher query string

            dataset:
                *Deprecated. Please provide your dataset as a Client constructor argument.*
                
                Which neuprint dataset to query against.
                If None provided, the client's default dataset is used.

            format:
                Either ``'pandas'`` or ``'json'``.
                Whether to load the results into a ``pandas.DataFrame``,
                or return the server's raw JSON response as a Python ``dict``.
        
        Returns:
            Either json or DataFrame, depending on ``format``.
        """
        url = f"{self.server}/api/custom/custom"
        return self._fetch_cypher(url, cypher, dataset, format)
    

    def _fetch_cypher(self, url, cypher, dataset, format='pandas'):
        """
        Fetch cypher from an endpoint.
        Called by fetch_custom and by Transaction queries.
        """
        assert format in ('json', 'pandas')
        
        if set("‘’“”").intersection(cypher):
            msg = ("Your cypher query contains 'smart quotes' (e.g. ‘foo’ or “foo”),"
                   " which are not valid characters in cypher."
                   " Please replace them with ordinary quotes (e.g. 'foo' or \"foo\").\n"
                   "Your query was:\n"
                   + cypher)
            raise RuntimeError(msg)
        
        dataset = dataset or self.dataset
        
        cypher = indent(dedent(cypher), '    ')
        logger.debug(f"Performing cypher query against dataset '{dataset}':\n{cypher}")
        
        result = self._fetch_json(url,
                                  json={"cypher": cypher, "dataset": dataset},
                                  ispost=True)

        if format == 'json':
            return result

        df = pd.DataFrame(result['data'], columns=result['columns'])
        return df


    ##
    ## API-META
    ##

    def fetch_available(self):
        """
        Fetch the list of REST API endpoints supported by the server.
        """
        return self._fetch_json(f"{self.server}/api/available")


    def fetch_help(self):
        """
        Fetch auto-generated REST API documentation, as YAML text.
        """
        return self._fetch_raw(f"{self.server}/api/help/swagger.yaml").decode('utf-8')


    def fetch_server_info(self):
        """
        Returns whether or not the server is public.
        """
        return self._fetch_json(f"{self.server}/api/serverinfo")['IsPublic']


    def fetch_version(self):
        """
        Returns the version of the ``neuPrintHTTP`` server.
        """
        return self._fetch_json(f"{self.server}/api/version")['Version']


    ##
    ## DB-META
    ##

    def fetch_database(self):
        """
        Fetch the address of the neo4j database that the neuprint server is using.
        """
        return self._fetch_json(f"{self.server}/api/dbmeta/database")


    def fetch_datasets(self):
        """
        Fetch basic information about the available datasets on the server.
        """
        return self._fetch_json(f"{self.server}/api/dbmeta/datasets")


    def fetch_instances(self):
        """
        Fetch secondary data instances avaiable through neupint http
        """
        return self._fetch_json(f"{self.server}/api/dbmeta/instances")

    
    def fetch_db_version(self):
        """
        Fetch the database version
        """
        return self._fetch_json(f"{self.server}/api/dbmeta/version")['Version']


    ##
    ## USER
    ##

    def fetch_profile(self):
        """
        Fetch basic information about your user profile,
        including your access level.
        """
        return self._fetch_json(f"{self.server}/profile")


    def fetch_token(self):
        """
        Fetch your user authentication token.
        
        Note:
            This method just echoes the token back to you for debug purposes.
            To obtain your token for the first time, use the neuprint explorer
            web UI to login and obtain your token as explained elsewhere in
            this documentation.
        """
        return self._fetch_json(f"{self.server}/token")['token']

    
    ##
    ## Cached
    ##
    
    def fetch_daily_type(self, format='pandas'):
        """
        Return information about today's cell type of the day.
        
        The server updates the completeness numbers each day. A different
        cell type is randomly picked and an exemplar is chosen
        from this type.

        Returns:
            If ``format='json'``, a dictionary is returned with keys
            ``['info', 'connectivity', 'skeleton']``.
            If ``format='pandas'``, three values are returned:
            ``(info, connectivity, skeleton)``, where ``connectivity``
            and ``skeleton`` are DataFrames.
        """
        assert format in ('json', 'pandas')
        url = f"{self.server}/api/cached/dailytype?dataset={self.dataset}"
        result = self._fetch_json(url, ispost=False)
        if format == 'json':
            return result
        
        conn_df = pd.DataFrame(result['connectivity']['data'],
                               columns=result['connectivity']['columns']) 
        skel_df = pd.DataFrame(result['skeleton']['data'],
                               columns=result['skeleton']['columns'])

        return result['info'], conn_df, skel_df

    
    def fetch_roi_completeness(self, format='pandas'):
        """
        Fetch the pre-computed traced "completeness" statistics
        for each primary ROI in the dataset.
        
        The completeness statistics indicate how many synapses
        belong to Traced neurons.
        
        Note:
            These results are not computed on-the-fly.
            They are computed periodically and cached.
        """
        assert format in ('json', 'pandas')
        url = f"{self.server}/api/cached/roicompleteness?dataset={self.dataset}"
        result = self._fetch_json(url, ispost=False)
        if format == 'json':
            return result
        
        df = pd.DataFrame(result['data'], columns=result['columns'])
        return df


    def fetch_roi_connectivity(self, format='pandas'):
        """
        Fetch the pre-computed connectivity statistics
        between primary ROIs in the dataset.
        
        Note:
            These results are not computed on-the-fly.
            They are computed periodically and cached.
        """
        assert format in ('json', 'pandas')
        url = f"{self.server}/api/cached/roiconnectivity?dataset={self.dataset}"
        result = self._fetch_json(url, ispost=False)
        if format == 'json':
            return result
        
        # Example result:
        # {
        #    "roi_names": [['ME(R)', "a'L(L)", 'aL(L)', ...]],
        #    "weights": {
        #       'EPA(R)=>gL(L)': {'count': 7, 'weight': 1.253483174941712},
        #       'EPA(R)=>gL(R)': {'count': 29, 'weight': 2.112117795621343},
        #       'FB=>AB(L)': {'count': 62, 'weight': 230.11732347331355},
        #       'FB=>AB(R)': {'count': 110, 'weight': 496.733276906109},
        #       ...
        #    }
        # }
        
        weights = [(*k.split('=>'), v['count'], v['weight']) for k,v in result["weights"].items()]
        df = pd.DataFrame(weights, columns=['from_roi', 'to_roi', 'count', 'weight'])
        return df


    ##
    ## ROI MESHES
    ##
    def fetch_roi_mesh(self, roi, export_path=None):
        """
        Fetch a mesh for the given ROI, in ``.obj`` format.
        
        Args:
            roi:
                Name of an ROI
            export_path:
                Optional. Writes the ``.obj`` file to the given path.
            
        Returns:
            bytes
            The contents of the fetched ``.obj`` mesh file.

        Note:
            ROI meshes are intended for visualization only.
            (They are not suitable for quantitative analysis.)
        """
        url = f"{self.server}/api/roimeshes/mesh/{self.dataset}/{roi}"
        data = self._fetch_raw(url, ispost=False)
        
        if export_path:
            with open(export_path, 'wb') as f:
                f.write(data)
        return data
        
    
    ##
    ## SKELETONS
    ##
    def fetch_skeleton(self, body, format='swc', export_path=None):
        """
        Fetch the skeleton for a neuron or segment.
        
        Args:
            body:
                int. A neuron or segment ID
            
            format:
                Either 'swc' (a text format), 'json', or 'pandas'.
            
            export_path:
                Optional. Writes the ``.swc`` file to disk.
        
        Returns:
            Either a string (swc), dict (json), or a DataFrame (pandas). 
        """
        assert format in ('swc', 'json', 'pandas'), \
            f'Invalid format: {format}'
        
        assert not export_path or format == 'swc', \
            "Only the swc format can be exported to disk."
        
        try:
            body = int(body)
        except ValueError:
            raise RuntimeError(f"Please pass an integer body ID, not '{body}'")

        url = f"{self.server}/api/skeletons/skeleton/{self.dataset}/{body}"
        if format == 'swc':
            url += '?format=swc'
            swc_text = self._fetch_raw(url, ispost=False).decode('utf-8')
            if export_path:
                with open(export_path, 'w') as f:
                    f.write(swc_text)
            return swc_text

        result = self._fetch_json(url, ispost=False)
        if format == 'json':
            return result

        df = pd.DataFrame(result['data'], columns=result['columns'])
        return df


    ##
    ## RAW KEY-VALUE
    ##
    def fetch_raw_keyvalue(self, instance, key):
        """
        Fetch a value from the ``neuprintHTTP`` server.
        The data address is given by both the instance name and key.
        (For admins and experts only.)
        """
        url = f"{self.server}/api/raw/keyvalue/key/{instance}/{key}"
        return self._fetch_raw(url, ispost=False)
        

    def post_raw_keyvalue(self, instance, key, value):
        """
        Post a value from the ``neuprintHTTP`` server.
        The data address is given by both the instance name and key.
        (For admins and experts only.)
        """
        assert isinstance(value, bytes)
        url = f"{self.server}/api/raw/keyvalue/key/{instance}/{key}"
        r = self.session.post(url, data=value, verify=self.verify)
        r.raise_for_status()
