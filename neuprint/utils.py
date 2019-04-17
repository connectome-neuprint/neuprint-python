import sys
from collections.abc import Iterable

from .client import Client

def eval_client(x=None):
    """ If x is not a client, will try returning global client.
    """

    if isinstance(x, Client):
        return x
    elif 'NEUPRINT_CLIENT' in sys.modules:
        return sys.modules['NEUPRINT_CLIENT']
    else:
        raise ValueError('No client found')


def make_iterable(x):
    """ Forces x into iterable """
    if isinstance(x, Iterable) and not isinstance(x, str):
        return x
    else:
        return [x]

def parse_properties(props, placeholder):
    """ Parses list of properties and returns a RETURN string."""
    props = props if isinstance(props, list) else list(props)

    cypher = []
    for p in props:
        if p == 'hasSkeleton':
            cypher.append('exists(({})-[:Contains]->(:Skeleton)) AS hasSkeleton'.format(placeholder))
        else:
            cypher.append('{0}.{1} AS {1}'.format(placeholder, p))

    return ','.join(cypher)