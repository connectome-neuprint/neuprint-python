import os
import platform

from .client import Client, default_client, set_default_client
from .segmentcriteria import SegmentCriteria
from .synapsecriteria import SynapseCriteria
from .queries import *
from .utils import merge_neuron_properties, connection_table_to_matrix

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

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
