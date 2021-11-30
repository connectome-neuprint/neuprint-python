from ..client import inject_client


@inject_client
def fetch_custom(cypher, dataset="", format='pandas', *, client=None):
    '''
    Make a custom cypher query.

    Alternative form of :py:meth:`.Client.fetch_custom()`, as a free function.
    That is, ``fetch_custom(..., client=c)`` is equivalent to ``c.fetch_custom(...)``.

    If ``client=None``, the default ``Client`` is used
    (assuming you have created at least one ``Client``.)

    Args:
        cypher:
            A cypher query string

        dataset:
            *Deprecated. Please provide your dataset as a Client constructor argument.*

            Which neuprint dataset to query against.
            If None provided, the client's default dataset is used.

        format:
            Either 'pandas' or 'json'.
            Whether to load the results into a pandas DataFrame,
            or return the server's raw json response as a Python dict.

        client:
            If not provided, the global default :py:class:`.Client` will be used.

    Returns:
        Either json or DataFrame, depending on ``format``.

    .. code-block:: ipython

        In [4]: from neuprint import fetch_custom
           ...:
           ...: q = """\\
           ...: MATCH (n:Neuron)
           ...: WHERE n.bodyId = 5813027016
           ...: RETURN n.type, n.instance
           ...: """
           ...: fetch_custom(q)
        Out[4]:
          n.type      n.instance
        0   FB4Y  FB4Y(EB/NO1)_R
    '''
    return client.fetch_custom(cypher, dataset, format)


@inject_client
def fetch_meta(*, client=None):
    """
    Fetch the dataset metadata.
    Parses json fields as needed.

    Returns:
        dict

    Example

    .. code-block:: ipython

        In [1]: from neuprint import fetch_meta

        In [2]: meta = fetch_meta()

        In [3]: list(meta.keys())
        Out[3]:
        ['dataset',
         'info',
         'lastDatabaseEdit',
         'latestMutationId',
         'logo',
         'meshHost',
         'neuroglancerInfo',
         'neuroglancerMeta',
         'postHPThreshold',
         'postHighAccuracyThreshold',
         'preHPThreshold',
         'primaryRois',
         'roiHierarchy',
         'roiInfo',
         'statusDefinitions',
         'superLevelRois',
         'tag',
         'totalPostCount',
         'totalPreCount',
         'uuid']
    """
    q = """\
        MATCH (m:Meta)
        WITH m as m,
             apoc.convert.fromJsonMap(m.roiInfo) as roiInfo,
             apoc.convert.fromJsonMap(m.roiHierarchy) as roiHierarchy,
             apoc.convert.fromJsonMap(m.neuroglancerInfo) as neuroglancerInfo,
             apoc.convert.fromJsonList(m.neuroglancerMeta) as neuroglancerMeta,
             apoc.convert.fromJsonMap(m.statusDefinitions) as statusDefinitions
        RETURN m as meta, roiInfo, roiHierarchy, neuroglancerInfo, neuroglancerMeta, statusDefinitions
    """
    df = client.fetch_custom(q)
    meta = df['meta'].iloc[0]
    meta.update(df.drop(columns='meta').iloc[0].to_dict())
    return meta
