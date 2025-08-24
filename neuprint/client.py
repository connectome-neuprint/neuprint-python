'''
All communication to the neuPrint server is peformed using a
:py:class:`Client` object, which holds your authorization
credentials, the dataset name to use, and other connection settings.
(To obtain your authorization credentials, see the :ref:`quickstart`.)


For commonly used queries, see :ref:`queries`.
Or you can implement your own cypher queries using :py:func:`.fetch_custom()`.

Example:

    .. code-block:: ipython

        In [1]: from neuprint import Client, fetch_custom, fetch_neurons

        In [2]: # Create a default client.
           ...: # It will be implicitly used for all subsequent queries as
           ...: long as it remains the only client you've created.
           ...: c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1')

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
import weakref
import functools
import threading
import collections
from textwrap import dedent, indent

import pandas as pd
import pyarrow as pa

from functools import lru_cache
from packaging import version

import urllib3
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning

from requests import Session, RequestException, HTTPError
from requests.adapters import HTTPAdapter

# ujson is faster than Python's builtin json module
import ujson

logger = logging.getLogger(__name__)

# These hold weak references
DEFAULT_NEUPRINT_CLIENT = lambda: None
USER_NEUPRINT_CLIENTS = set()

# This holds real references
DEFAULT_NEUPRINT_CLIENT_THREAD_COPIES = {}

_global_client_lock = threading.RLock()


class NeuprintTimeoutError(HTTPError):
    pass


def default_client():
    """
    Obtain the default Client object to use.
    This function returns a separate copy of the
    default client for each thread (and process).

    There's usually no need to call this function.
    It is automatically called by all query functions if
    you haven't passed in an explict `client` argument.
    """
    with _global_client_lock:
        default = DEFAULT_NEUPRINT_CLIENT()
        if default is None:
            clients = {c() for c in USER_NEUPRINT_CLIENTS if c()}
            if len(clients) == 0:
                raise RuntimeError(
                    "No default Client has been set yet because you haven't yet created a Client.\n"
                )
            if len(clients) > 1:
                raise RuntimeError(
                    "Currently more than one Client exists, so neither was automatically chosen as the default.\n"
                    "You must explicitly pass a client to query functions, or explicitly call set_default_client().\n"
                    f"Currently {len(clients)} clients exist: {clients}"
            )
            if len(clients) == 1:
                raise RuntimeError(
                    "No default Client has been set. One client does exist already, "
                    "which can be made the default via set_default_client()."
                )

        thread_id = threading.current_thread().ident
        pid = os.getpid()

        try:
            c = DEFAULT_NEUPRINT_CLIENT_THREAD_COPIES[(thread_id, pid)]
        except KeyError:
            c = copy.deepcopy(default)
            DEFAULT_NEUPRINT_CLIENT_THREAD_COPIES[(thread_id, pid)] = c
        return c


def set_default_client(client):
    """
    Set (or overwrite) the default Client.

    There's usually no need to call this function.
    It's is automatically called when your first
    ``Client`` is created, but you can call it again
    to replace the default.
    """
    if client is None:
        clear_default_client()
        return

    global DEFAULT_NEUPRINT_CLIENT  # noqa
    DEFAULT_NEUPRINT_CLIENT = weakref.ref(client)

    thread_id = threading.current_thread().ident
    pid = os.getpid()

    if DEFAULT_NEUPRINT_CLIENT_THREAD_COPIES.get((thread_id, pid), None) == client:
        return

    with _global_client_lock:
        DEFAULT_NEUPRINT_CLIENT_THREAD_COPIES.clear()

        # We temporarily store the original object before performing the deepcopy below.
        # The current function is called during depickling, so this avoids infinite recursion.
        DEFAULT_NEUPRINT_CLIENT_THREAD_COPIES[(thread_id, pid)] = client

        # We exclusively store *copies* in this dict,
        # to ensure that the DEFAULT_NEUPRINT_CLIENT weakref becomes
        # invalid if the user deletes their Client.
        DEFAULT_NEUPRINT_CLIENT_THREAD_COPIES[(thread_id, pid)] = copy.deepcopy(client)

        # Sadly, we must set this again because the deepcopy above
        # triggered registration of the client and cleared the default.
        # This is all a mess and needs to be reworked.
        DEFAULT_NEUPRINT_CLIENT = weakref.ref(client)


def clear_default_client():
    """
    Unset the default Client, leaving no default in place.
    """
    global DEFAULT_NEUPRINT_CLIENT  # noqa
    with _global_client_lock:
        DEFAULT_NEUPRINT_CLIENT = lambda: None  # noqa
        DEFAULT_NEUPRINT_CLIENT_THREAD_COPIES.clear()


def list_all_clients():
    """
    List all ``Client`` objects in the program.
    """
    return {c() for c in USER_NEUPRINT_CLIENTS if c()}


def _register_client(client):
    """
    Register the client in our global list of client weakrefs,
    and also set it as the default client if it happens to be the
    ONLY client in existence.

    We use weak references instead of regular references to leave
    the user in control of which clients "still exist".

    For example, the user should be able to do this:

        .. code-block:: python

            c = Client('neuprint.janelia.org', 'hemibrain:v1.2.1')
            n1, w1 = fetch_neurons(..., client=None)

            c = Client('neprint.janelia.org', 'manc:v1.0')
            n2, w2 = fetch_neurons(..., client=None)

    And since the first Client is deallocated, the second one
    becomes the default without issues.
    """
    added_client = False
    with _global_client_lock:
        clients = copy.copy(USER_NEUPRINT_CLIENTS)
        # Housekeeping: drop invalid references
        clients = {c for c in clients if c()}
        w = weakref.ref(client)
        if w not in clients:
            clients.add(w)
            added_client = True
        USER_NEUPRINT_CLIENTS.clear()
        USER_NEUPRINT_CLIENTS.update(clients)

    if not added_client:
        return

    # If there weren't any other clients,
    # then the new one becomes the default.
    if len(clients) == 1:
        set_default_client(client)
    else:
        # Otherwise, the default is cleared.
        # If the user really wants to pick a default client,
        # they can call set_default_client() explicitly.
        clear_default_client()


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
               ...: from neuprint import fetch_neurons, NeuronCriteria as NC
               ...:
               ...: setup_debug_logging()
               ...: neuron_df, roi_df = fetch_neurons(NC(type='MBON.*', rois=['MB(R)']))
            [2020-01-30 08:48:20,367] DEBUG Performing cypher query against dataset 'hemibrain:v1.2.1':
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

    enable_debug_logging()


def enable_debug_logging():
    logger.setLevel(logging.DEBUG)


def disable_debug_logging():
    logger.setLevel(logging.INFO)


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
                raise  # Nothing to add to the exception.

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
                msg += "\nReturned Error"
                if hasattr(ex.response, 'status_code'):
                    msg += f" ({ex.response.status_code})"
                if ex.response.content:
                    try:
                        err = ex.response.json()['error']
                        msg += f":\n\n{err}"
                    except Exception:  # noqa
                        pass

            if ex.response and 'timeout' in ex.response.content.decode('utf-8').lower():
                new_ex = NeuprintTimeoutError(msg, *ex.args[1:], response=ex.response, request=ex.request)
            else:
                new_ex = copy.copy(ex)
                new_ex.args = (msg, *ex.args[1:])

            # In case this decorator is used twice in a nested call,
            # mark it as already modified it doesn't get modified twice.
            new_ex.response_content_appended = True
            raise new_ex from ex
    return wrapper


class Client:
    """
    Client object for interacting with the neuprint database.
    """
    def __init__(self, server, dataset=None, token=None, verify=True, progress=True):
        """
        When you create the first ``Client``, it becomes the default
        ``Client`` to be used with all ``neuprint-python`` functions
        if you don't explicitly specify one.
        But if you create multiple ``Client`` objects, the default client
        is cleared and you must explicitly pass a ``client`` parameter to all
        query functions.

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

            progress:
                If ``True`` (default), show progress bars for long queries.

        """
        self.progress = progress

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
        self.token = token

        self.session = Session()
        self.session.headers.update({"Authorization": "Bearer " + token,
                                     "Content-type": "application/json"})

        # If the connection fails, retry a couple times.
        retries = Retry(connect=2, backoff_factor=0.1)
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

        self.verify = verify
        if not verify:
            urllib3.disable_warnings(InsecureRequestWarning)

        try:
            self.dataset = self._select_dataset(dataset, reload_cache=False)
        except RuntimeError:
            # One more try, in case we were using an outdated dataset list.
            self.dataset = self._select_dataset(dataset, reload_cache=True)

        _register_client(self)

    def _select_dataset(self, dataset, reload_cache):
        all_datasets = [*self.fetch_datasets(reload_cache).keys()]
        if len(all_datasets) == 0:
            raise RuntimeError(f"The neuprint server {self.server} has no datasets!")

        if len(all_datasets) == 1 and not dataset:
            logger.info(f"Initializing neuprint.Client with dataset: {dataset}")  # noqa
            return all_datasets[0]
        elif dataset in all_datasets:
            return dataset

        raise RuntimeError(f"Dataset '{dataset}' does not exist on"
                            f" the neuprint server ({self.server}).\n"
                            f"Available datasets: {all_datasets}")

    def __repr__(self):
        s = f'Client("{self.server}", "{self.dataset}"'
        if not self.verify:
            s += ", verify=False"
        s += ")"
        return s

    def __eq__(self, other):
        return (
            other and
            self.server == other.server and  # noqa
            self.dataset == other.dataset and  # noqa
            self.token == other.token and  # noqa
            self.verify == other.verify
        )

    def __hash__(self):
        return hash((self.server, self.dataset, self.token, self.verify))

    def __setstate__(self, state):
        self.__dict__.update(state)
        _register_client(self)

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

    def _fetch_arrow(self, url, json=None, ispost=False):
        r = self._fetch(url, json=json, ispost=ispost)
        content_type = r.headers.get('Content-Type', '')
        if 'application/vnd.apache.arrow.stream' not in content_type:
            raise Exception(f"Expected Arrow stream content type but got: {content_type}")
        reader = pa.ipc.open_stream(pa.py_buffer(r.content))
        return reader.read_all().to_pandas(maps_as_pydicts='strict')

    ##
    ## Cached properties
    ##

    @property
    @lru_cache
    def meta(self):
        from .queries.general import fetch_meta
        return fetch_meta(client=self)

    @property
    @lru_cache
    def primary_rois(self):
        return sorted(self.meta['primaryRois'])

    @property
    @lru_cache
    def all_rois(self):
        from .queries.rois import _all_rois_from_meta
        return _all_rois_from_meta(self.meta)

    ##
    ## CUSTOM QUERIES
    ##
    ## Note: Transaction queries are not implemented here.  See admin.py
    ##

    def fetch_custom(self, cypher, dataset="", format='pandas', use_arrow=False):  # noqa
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

            use_arrow:
                Behind the scenes, fetch data from the server using Arrow IPC instead of JSON.

                If False, use JSON.
                If True, use Arrow or raise an error if the server does not support it.
                If None, use Arrow if the server supports it, otherwise use JSON.

                Note:
                    At the time of this writing, neuprintHTTP performs slightly worse with the
                    Arrow IPC format than it does with JSON, which is why the default is JSON.

        Returns:
            json or DataFrame, depending on ``format``.
        """
        assert format in ('json', 'pandas')
        assert use_arrow in (True, False, None)
        dataset = dataset or self.dataset
        server_handles_arrow = self.arrow_endpoint()

        if format == 'json' and use_arrow:
            raise RuntimeError("Returning JSON via Arrow is not supported.")

        if use_arrow and not server_handles_arrow:
            raise RuntimeError("Cannot use arrow: Server does not support Arrow IPC.")

        if set("‘’“”").intersection(cypher):
            msg = ("Your cypher query contains 'smart quotes' (e.g. ‘foo’ or “foo”),"
                   " which are not valid characters in cypher."
                   " Please replace them with ordinary quotes (e.g. 'foo' or \"foo\").\n"
                   "Your query was:\n" + cypher)
            raise RuntimeError(msg)

        cypher = indent(dedent(cypher), '    ')
        logger.debug(f"Performing cypher query against dataset '{dataset}':\n{cypher}")

        if format == 'pandas' and server_handles_arrow and use_arrow is not False:
            return self._fetch_arrow(
                f"{self.server}/api/custom/arrow",
                {"cypher": cypher, "dataset": dataset},
                True
            )

        response = self._fetch_json(
            f"{self.server}/api/custom/custom",
            {"cypher": cypher, "dataset": dataset},
            True
        )

        if format == 'json':
            return response

        return pd.DataFrame(response['data'], columns=response['columns'])


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

        Returns:
            str: The server version as a string.

        Raises:
            HTTPError: If the version endpoint is not found or returns an error.
            KeyError: If the response doesn't contain a 'Version' key.
            Exception: For other unexpected errors.
        """
        try:
            response = self._fetch_json(f"{self.server}/api/version")
            return response['Version']
        except (HTTPError, KeyError, Exception) as e:
            # Let the caller handle the exception
            raise

    def arrow_endpoint(self):
        """
        Checks if the neuPrintHTTP server version supports Arrow IPC via HTTP.

        Returns:
            bool: True if the server version is 1.7.3 or higher, False otherwise.
        """
        try:
            version_str = self.fetch_version()
            if not version_str:
                return False

            # Parse semantic version
            server_version = version.parse(version_str)
            min_version = version.parse('1.7.3')

            return server_version >= min_version
        except (HTTPError, KeyError, Exception):
            # If we can't determine the version for any reason, default to False
            return False

    @lru_cache
    def fetch_neuron_keys(self):
        """
        Returns all available :Neuron properties in the database. Cached.
        """
        b = "MATCH (n:Meta) RETURN n.neuronProperties"
        df_results = self.fetch_custom(b)
        neuron_props_val = df_results.iloc[0, 0]
        if neuron_props_val is None:
            # Fetch available keys
            c = """
            MATCH (n :Neuron) UNWIND KEYS(n) AS k RETURN DISTINCT k AS neuron_fields
            """
            raw = self.fetch_custom(c, format='json')
            return [r[0] for r in raw['data']]
        else:
            # use neuronProperties to report neuron keys
            neuron_props_val = df_results.iloc[0, 0]
            neuron_props_json = ujson.loads(neuron_props_val)
            neuron_props = list(neuron_props_json.keys())
            return neuron_props

    @lru_cache
    def fetch_synapse_nt_keys(self):
        """
        Returns :Synapse properties related to neurotransmitters, sorted. Cached.

        The properties may be stored in the :Meta node, or they may be
        queried from the database directly.
        """

        # first, check the :Meta node; it'll have a dict of {property: property type}
        b = "MATCH (n:Meta) RETURN n.ntSynapseProperties"
        df_results = self.fetch_custom(b)
        synapse_props_val = df_results.iloc[0, 0]
        if synapse_props_val is not None:
            synapse_props_json = ujson.loads(synapse_props_val)
            synapse_props = list(synapse_props_json.keys())
            return sorted(synapse_props)
        else:
            # fallback: query the database and see actually exists;
            #   note all "pre" values have the same props
            q = """
                MATCH (s: Synapse {type:"pre"}) 
                WITH [k IN keys(s) WHERE k STARTS WITH 'nt'] AS nt_keys
                RETURN nt_keys
                LIMIT 1
                """
            json_data = self.fetch_custom(q, format='json')
            if json_data['data']:
                return sorted(json_data['data'][0][0])
            else:
                logger.warning("No synapse neurotransmitter keys found in the database.")
                return []

    ##
    ## DB-META
    ##

    def fetch_database(self):
        """
        Fetch the address of the neo4j database that the neuprint server is using.
        """
        return self._fetch_json(f"{self.server}/api/dbmeta/database")

    DATASETS_CACHE = {}

    def fetch_datasets(self, reload_cache=False):
        """
        Fetch basic information about the available datasets on the server.

        Args:
            reload_cache:
                The result from each unique neuprint server is cached locally
                and re-used by all Clients, but you can invalidate the entire
                cache by setting reload_cache to True, causing it to be repopulated
                during this call.

        Returns:
            dict, keyed by dataset name
        """
        if reload_cache:
            Client.DATASETS_CACHE.clear()

        try:
            return Client.DATASETS_CACHE[self.server]
        except KeyError:
            datasets = self._fetch_json(f"{self.server}/api/dbmeta/datasets")
            Client.DATASETS_CACHE[self.server] = datasets
            return datasets

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
    def fetch_skeleton(self, body, heal=False, export_path=None, format='pandas', with_distances=False):
        """
        Fetch the skeleton for a neuron or segment.

        Args:

            body (int):
                A neuron or segment ID

            heal (bool):
                If ``True`` and the skeleton is fragmented, 'heal' it by connecting
                its fragments into a single tree. The fragments are joined by
                selecting the minimum spanning tree after joining all fragments
                via their pairwise nearest neighbors. See :py:func:`.heal_skeleton()`
                for more details.

                If you want the healing procedure to refrain from connecting very
                distant fragments, set ``heal`` to a maximum allowed distance,
                e.g. ``heal=1000.0``

            format (str):
                Either 'pandas', 'swc' (similar to CSV), or 'nx' (``networkx.DiGraph``).

            export_path (str):
                Optional. Writes the ``.swc`` file to disk.
                (SWC format is written, regardless of the returned ``format``.)

            with_distances:
                Only valid when format is ``pandas`` or ``nx``.
                If True, a 'distance' column (or edge attribute) will be added
                to the dataframe (or nx.Graph), indicating the distances from each
                node to its parent node.
                In DataFrame results, root nodes will be assigned a distance of ``np.inf``.
                Distances are computed AFTER healing is performed.
                Distances will not be present in any exported SWC file.

        Returns:

            Either a string (swc), a DataFrame (pandas), or ``networkx.DiGraph`` (nx).

        See also:

            - :py:func:`.heal_skeleton()`
            - :py:func:`.skeleton_df_to_nx()`
            - :py:func:`.skeleton_df_to_swc()`
        """
        from .skeleton import skeleton_df_to_nx, heal_skeleton, skeleton_df_to_swc, skeleton_swc_to_df, calc_segment_distances

        try:
            body = int(body)
        except ValueError:
            raise RuntimeError(f"Please pass an integer body ID, not '{body}'")

        assert format in ('swc', 'pandas', 'nx'), f'Invalid format: {format}'
        assert not with_distances or format in ('pandas', 'nx'), \
            f"The with_distances option can only be used with the 'pandas' or 'nx' output formats, not {format}"

        url = f"{self.server}/api/skeletons/skeleton/{self.dataset}/{body}?format=swc"
        swc = self._fetch_raw(url, ispost=False).decode('utf-8')

        if heal or format != 'swc':
            df = skeleton_swc_to_df(swc)

        if heal:
            df = heal_skeleton(df, heal, -1)
            if export_path or format == 'swc':
                swc = skeleton_df_to_swc(df)

        if export_path:
            with open(export_path, 'w') as f:
                f.write(swc)

        if format == 'swc':
            return swc

        if format == 'pandas':
            if with_distances:
                df['distance'] = calc_segment_distances(df)
            return df

        if format == 'nx':
            return skeleton_df_to_nx(df, with_distances=with_distances)

        raise AssertionError('Should not get here.')

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
