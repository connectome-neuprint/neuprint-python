from collections.abc import Iterable

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
            cypher.append(f'exists(({placeholder})-[:Contains]->(:Skeleton)) AS hasSkeleton')
        else:
            cypher.append(f'{placeholder}.{p} AS {p}')

    return ','.join(cypher)