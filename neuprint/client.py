# -*- coding: utf-8 -*-
import os
import copy
import json
import inspect
import functools
import threading
import collections

import pandas as pd

import urllib3
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning

from requests import Session, RequestException, HTTPError
from requests.adapters import HTTPAdapter


try:
    # ujson is faster than Python's builtin json module;
    # use it if the user happens to have it installed.
    import ujson
    _use_ujson = True
except ImportError:
    _use_ujson = False


DEFAULT_NEUPRINT_CLIENT = None
NEUPRINT_CLIENTS = {}


def default_client():
    """
    Obtain the default Client object to use.
    This function returns a separate copy of the
    default client for each thread (and process).
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
    """
    global NEUPRINT_CLIENTS
    global DEFAULT_NEUPRINT_CLIENT

    thread_id = threading.current_thread().ident
    pid = os.getpid()

    DEFAULT_NEUPRINT_CLIENT = client
    NEUPRINT_CLIENTS.clear()
    NEUPRINT_CLIENTS[(thread_id, pid)] = client


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
    return wrapper


@inject_client
def fetch_custom(cypher, dataset="", format='pandas', *, client=None):
    """
    Alternative form of Client.fetch_custom(), as a free function.
    That is, ``fetch_custom(..., client=c)`` is equivalent to ``c.fetch_custom(...)``.
    
    Args:
        cypher:
            A cypher query string
        dataset:
            Which neuprint dataset to query against.
            If None provided, the client's default dataset is used.
            If the client has no default dataset configured,
            the server will use its own default.
        format:
            Either 'pandas' or 'json'.
            Whether to load the results into a pandas DataFrame,
            or return the server's raw JSON response as a Python dict.
        client:
            If not provided, the global default ``Client`` will be used.
    
    Returns:
        Either json or DataFrame, depending on ``format``.
    """
    return client.fetch_custom(cypher, dataset, format)


class Client:
    '''
    Used for all queries against neuprint.
    Holds your authorization credentials, the dataset name to use,
    and other connection settings.
    
    Most ``neuprint-python`` functions do not require you to explicitly
    provide a Client object to use. Instead, the first ``Client`` you
    create will be stored as the default ``Client`` to be used with all
    ``neuprint-python`` functions if you don't explicitly specify one.
    
    Example:
    
        .. code-block:: python
        
            # Create a Client to be used globally.
            c = Client('neuprint.janelia.org')
            
            # Subsequent calls use the global client implicitly.
            fetch_custom("""\
                MATCH (n: Neuron)
                WHERE n.status = "Traced"
                RETURN n.bodyId
            """)
    '''
    def __init__(self, server, token=None, verify=True, dataset=None):
        """
        Client constructor.
        
        Args:
            server:
                URL of neuprintHttp server

            token:
                neuPrint token. Either pass explitily as an argument or set
                as NEUPRINT_APPLICATION_CREDENTIALS environment variable.
                Your token can be retrieved by clicking on your account in
                the NeuPrint web interface.

            verify:
                If True (default), enforce signed credentials.

            dataset:
                The dataset to run all queries against, e.g. 'hemibrain'.
                If not provided, the server will use a default dataset for
                all queries.
        """
        if token is None:
            token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS')

        if token is None:
            raise RuntimeError("No token provided. Please provide one or set NEUPRINT_APPLICATION_CREDENTIALS")

        if ':' in token:
            try:
                token = json.loads(token)['token']
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

        self.server = server

        self.session = Session()
        self.session.headers.update({"Authorization": "Bearer " + token,
                                     "Content-type": "application/json"})

        # If the connection fails, retry a couple times.
        retries = Retry(connect=2, backoff_factor=0.1)
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

        self.verbose = False
        self.verify = verify
        self.current_transaction = None
        self.dataset = dataset or ""

        if not verify:
            urllib3.disable_warnings(InsecureRequestWarning)

        # Set this as the default client if there isn't one already
        global DEFAULT_NEUPRINT_CLIENT
        if DEFAULT_NEUPRINT_CLIENT is None:
            set_default_client(self)


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
        if _use_ujson:
            return ujson.loads(self._fetch(url, json=json, ispost=ispost).content)
        else:
            return self._fetch(url, json=json, ispost=ispost).json()


    def fetch_help(self):
        return self._fetch_raw(f"{self.server}/api/help")


    def fetch_version(self):
        return self._fetch_json(f"{self.server}/api/version")


    def fetch_available(self):
        return self._fetch_json(f"{self.server}/api/available")


    def fetch_database(self):
        """ Fetch available datasets.
        """
        return self._fetch_json(f"{self.server}/api/dbmeta/database")


    def fetch_datasets(self):
        """ Fetch available datasets.
        """
        return self._fetch_json(f"{self.server}/api/dbmeta/datasets")


    def fetch_custom(self, cypher, dataset="", format='pandas'):
        """
        Query the neuprint server with a custom Cypher query.
        
        Args:
            cypher:
                A cypher query string

            dataset:
                Which neuprint dataset to query against.
                If None provided, the client's default dataset is used.
                If the client has no default dataset configured,
                the server will use its own default.

            format:
                Either 'pandas' or 'json'.
                Whether to load the results into a pandas DataFrame,
                or return the server's raw JSON response as a Python dict.
        
        Returns:
            Either json or DataFrame, depending on ``format``.
        """
        if set("‘’“”").intersection(cypher):
            msg = ("Your cypher query contains 'smart quotes' (e.g. ‘foo’ or “foo”),"
                   " which are not valid characters in cypher."
                   " Please replace them with ordinary quotes (e.g. 'foo' or \"foo\").\n"
                   "Your query was:\n"
                   + cypher)
            raise RuntimeError(msg)
        
        assert format in ('json', 'pandas')
        
        dataset = dataset or self.dataset
        
        result = self._fetch_json(f"{self.server}/api/custom/custom",
                                  json={"cypher": cypher, "dataset": dataset},
                                  ispost=True)

        if format == 'json':
            return result

        df = pd.DataFrame(result['data'], columns=result['columns'])
        return df


    def start_transaction(self, dataset):
        """Starts a transaction of several cypher queries.

        Note: admin permission only.  Setting the dataset is needed to choose
        the proper database.
        """

        # remove previous transaction
        try:
            if self.current_transaction is not None:
                self.kill_transaction()
        except:
            pass

        self.dataset = dataset
        result = self._fetch_json(f"{self.server}/api/raw/cypher/transaction", 
                json={"dataset": dataset}, ispost=True)
        self.current_transaction = result["transaction_id"]
        return


    def kill_transaction(self):
        """Kills (rolls back) transaction.

        Note: admin permission only.
        """
        if self.current_transaction is None:
            raise RuntimeError("no transaction was created")

        oldtrans = self.current_transaction
        self.current_transaction = None
        self._fetch_json(f"{self.server}/api/raw/cypher/transaction/{oldtrans}/kill", ispost=True)
        
        return


    def commit_transaction(self):
        """Commits transaction.

        Note: admin permission only.
        """
        if self.current_transaction is None:
            raise RuntimeError("no transaction was created")
        oldtrans = self.current_transaction
        self.current_transaction = None

        self._fetch_json(f"{self.server}/api/raw/cypher/transaction/{oldtrans}/commit", ispost=True)

        return

    
    def query_transaction(self, cypher, format='pandas'):
        """ Make a custom cypher query (allows writes).

        Note: Admin permission only.  For this raw query, the dataset must be provided.
        """
        if self.current_transaction is None:
            raise RuntimeError("no transaction was created")
        
        if set("‘’“”").intersection(cypher):
            msg = ("Your cypher query contains 'smart quotes' (e.g. ‘foo’ or “foo”),"
                   " which are not valid characters in cypher."
                   " Please replace them with ordinary quotes (e.g. 'foo' or \"foo\").\n"
                   "Your query was:\n"
                   + cypher)
            raise RuntimeError(msg)
        
        assert format in ('json', 'pandas')
        result = self._fetch_json(f"{self.server}/api/raw/cypher/transaction/{self.current_transaction}/cypher",
                                  json={"cypher": cypher, "dataset": self.dataset},
                                  ispost=True)
        if format == 'json':
            return result

        df = pd.DataFrame(result['data'], columns=result['columns'])
        return df


