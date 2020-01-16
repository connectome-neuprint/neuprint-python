# -*- coding: utf-8 -*-
import copy
import json
import os
import sys
import platform

import pandas as pd

import urllib3
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning

from requests import Session, RequestException
from requests.adapters import HTTPAdapter

try:
    # ujson is faster than Python's builtin json module;
    # use it if the user happens to have it installed.
    import ujson
    _use_ujson = True
except ImportError:
    _use_ujson = False


# On Mac, requests uses a system library which is not fork-safe,
# so using multiprocessing results in segfaults such as the following:
#
#   File ".../lib/python3.7/urllib/request.py", line 2588 in proxy_bypass_macosx_sysconf
#   File ".../lib/python3.7/urllib/request.py", line 2612 in proxy_bypass
#   File ".../lib/python3.7/site-packages/requests/utils.py", line 745 in should_bypass_proxies
#   File ".../lib/python3.7/site-packages/requests/utils.py", line 761 in get_environ_proxies
#   File ".../lib/python3.7/site-packages/requests/sessions.py", line 700 in merge_environment_settings
#   File ".../lib/python3.7/site-packages/requests/sessions.py", line 524 in request
#   File ".../lib/python3.7/site-packages/requests/sessions.py", line 546 in get
# ...

# The workaround is to set a special environment variable
# to avoid the particular system function in question.
# Details here:
# https://bugs.python.org/issue30385
if platform.system() == "Darwin":
    os.environ["no_proxy"] = "*"



class Client:
    """ Holds your NeuPrint credentials and does the data fetching.

    Parameters
    ----------
    server :        str
                    URL of server.
    token :         str, optional
                    NeuPrint token. Either pass explitily as an argument or set
                    as NEUPRINT_APPLICATION_CREDENTIALS environment variable.
                    Your token can be retrieved by clicking on your account in
                    the NeuPrint web interface.
    set_global :    bool, optional
                    If True (default), will make this client global so that
                    you don't have to explicitly pass it to each function.
    verify :        If True (default), enforce signed credentials.
    """

    def __init__(self, server, token=None, set_global=True, verify=True):
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
            raise RuntimeError("Unknown protocol: {}".format(server.split('://')[0]))

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
        self.dataset = ""

        if not verify:
            urllib3.disable_warnings(InsecureRequestWarning)

        if set_global:
            self.make_global()

    def make_global(self):
        """Sets this variable as global by attaching it as sys.module"""
        sys.modules['NEUPRINT_CLIENT'] = self

    def _fetch(self, url, json=None, ispost=False):
        if self.verbose:
            print('url:', url)
            if json is not None:
                print('cypher:', json.get('cypher'))

        try:
            if ispost:
                r = self.session.post(url, json=json, verify=self.verify)
            else:
                assert json is None, "Can't provide a body via GET method"
                r = self.session.get(url, verify=self.verify)
            r.raise_for_status()
            return r
        except RequestException as ex:
            # If the error response had content (and it's not super-long),
            # show that in the traceback, too.  neuprint might provide a useful
            # error message in the response body.
            if (ex.response is not None or ex.request is not None):
                msg = ""
                if (ex.request is not None):
                    msg += "Error accessing {} {}\n".format(ex.request.method, ex.request.url)

                if (ex.response is not None and ex.response.content and len(ex.response.content) <= 1000):
                        msg += str(ex.args[0]) + "\n" + ex.response.content.decode('utf-8') + "\n"
                    

                if json is not None:
                    cypher = json.get('cypher')
                    if cypher:
                        msg += "\nCypher was:\n\n{}\n".format(cypher)
                
                if (ex.response is not None and ex.response.content):
                    try:
                        err = ex.response.json()['error']
                        msg += "\nReturned Error:\n\n{}".format(err)
                    except Exception:
                        pass

                new_ex = copy.copy(ex)
                new_ex.args = (msg, *ex.args[1:])
                raise new_ex from ex
            else:
                raise

    def _fetch_raw(self, url, json=None, ispost=False):
        return self._fetch(url, json=json, ispost=ispost).content

    def _fetch_json(self, url, json=None, ispost=False):
        if _use_ujson:
            return ujson.loads(self._fetch(url, json=json, ispost=ispost).content)
        else:
            return self._fetch(url, json=json, ispost=ispost).json()

    def fetch_help(self):
        return self._fetch_raw("{}/api/help".format(self.server))

    def fetch_version(self):
        return self._fetch_json("{}/api/version".format(self.server))

    def fetch_available(self):
        return self._fetch_json("{}/api/available".format(self.server))

    def fetch_database(self):
        """ Fetch available datasets.
        """
        return self._fetch_json("{}/api/dbmeta/database".format(self.server))

    def fetch_datasets(self):
        """ Fetch available datasets.
        """
        return self._fetch_json("{}/api/dbmeta/datasets".format(self.server))

    def fetch_custom(self, cypher, dataset="", format='pandas'):
        """ Fetch custom cypher.

        Note: if a dataset is not specified, the default database will be used
        and the caller must specify the dataset explicitly in the queries as needed.
        """
        if set("‘’“”").intersection(cypher):
            msg = ("Your cypher query contains 'smart quotes' (e.g. ‘foo’ or “foo”),"
                   " which are not valid characters in cypher."
                   " Please replace them with ordinary quotes (e.g. 'foo' or \"foo\").\n"
                   "Your query was:\n"
                   + cypher)
            raise RuntimeError(msg)
        
        assert format in ('json', 'pandas')
        result = self._fetch_json("{}/api/custom/custom".format(self.server),
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
        result = self._fetch_json("{}/api/raw/cypher/transaction".format(self.server), 
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
        self._fetch_json("{}/api/raw/cypher/transaction/{}/kill".format(self.server, oldtrans), ispost=True)
        
        return

    def commit_transaction(self):
        """Commits transaction.

        Note: admin permission only.
        """
        if self.current_transaction is None:
            raise RuntimeError("no transaction was created")
        oldtrans = self.current_transaction
        self.current_transaction = None

        self._fetch_json("{}/api/raw/cypher/transaction/{}/commit".format(self.server, oldtrans), ispost=True)

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
        result = self._fetch_json("{}/api/raw/cypher/transaction/{}/cypher".format(self.server, self.current_transaction),
                json={"cypher": cypher, "dataset": self.dataset}, ispost=True)
        if format == 'json':
            return result

        df = pd.DataFrame(result['data'], columns=result['columns'])
        return df


