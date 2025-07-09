"""
Utility functions for manipulating neuprint-python output.
"""
import re
import os
import sys
import inspect
import functools
import warnings
from textwrap import dedent
from collections.abc import Iterable, Iterator, Collection

import numpy as np
import pandas as pd
from requests import Session
import ujson


class NotNull:
    """Filter for existing properties.

    Translates to::

        WHERE neuron.{property} IS NOT NULL

    """


class IsNull:
    """Filter for missing properties.

    Translates to::

        WHERE neuron.{property} IS NULL

    """


CYPHER_KEYWORDS = [
    "CALL", "CREATE", "DELETE", "DETACH", "FOREACH", "LOAD", "MATCH", "MERGE", "OPTIONAL", "REMOVE", "RETURN", "SET", "START", "UNION", "UNWIND", "WITH",
    "LIMIT", "ORDER", "SKIP", "WHERE", "YIELD",
    "ASC", "ASCENDING", "ASSERT", "BY", "CSV", "DESC", "DESCENDING", "ON",
    "ALL", "CASE", "COUNT", "ELSE", "END", "EXISTS", "THEN", "WHEN",
    "AND", "AS", "CONTAINS", "DISTINCT", "ENDS", "IN", "IS", "NOT", "OR", "STARTS", "XOR",
    "CONSTRAINT", "CREATE", "DROP", "EXISTS", "INDEX", "NODE", "KEY", "UNIQUE",
    "INDEX", "JOIN", "SCAN", "USING",
    "FALSE", "NULL", "TRUE",
    "ADD", "DO", "FOR", "MANDATORY", "OF", "REQUIRE", "SCALAR"
]

# Technically this pattern is too strict, as it doesn't allow for non-ascii letters,
# but that's okay -- we just might use backticks a little more often than necessary.
CYPHER_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def cypher_identifier(name):
    """
    Wrap the given name in backticks if it wouldn't be a vlid cypher identifier without them.
    """
    if name.upper() in CYPHER_KEYWORDS or not CYPHER_IDENTIFIER_PATTERN.match(name):
        return f"`{name}`"
    return name


#
# Import the notebook-aware version of tqdm if
# we appear to be running within a notebook context.
#
try:
    import ipykernel.iostream
    if isinstance(sys.stdout, ipykernel.iostream.OutStream):
        from tqdm.notebook import tqdm

        try:
            import ipywidgets
            ipywidgets
        except ImportError:
            msg = dedent("""\

                Progress bar will not work well in the notebook without ipywidgets.
                Run the following commands (for notebook and jupyterlab users):

                    conda install -c conda-forge ipywidgets
                    jupyter nbextension enable --py widgetsnbextension
                    jupyter labextension install @jupyter-widgets/jupyterlab-manager

                ...and then reload your jupyter session, and restart your kernel.
            """)
            warnings.warn(msg)
    else:
        from tqdm import tqdm

except ImportError:
    from tqdm import tqdm


class tqdm(tqdm):
    """
    Same as tqdm, but auto-disable the progress bar if there's only one item.
    """
    def __init__(self, iterable=None, *args, disable=None, **kwargs):
        if disable is None:
            disable = (iterable is not None
                       and hasattr(iterable, '__len__')
                       and len(iterable) <= 1)

        super().__init__(iterable, *args, disable=disable, **kwargs)


def trange(*args, **kwargs):
    return tqdm(range(*args), **kwargs)


def UMAP(*args, **kwargs):
    """
    UMAP is an optional dependency, so this wrapper emits
    a nicer error message if it's not available.
    """
    try:
        from umap import UMAP
    except ImportError as ex:
        msg = (
            "The 'umap' dimensionality reduction package is required for some "
            "plotting functionality, but it isn't currently installed.\n\n"
            "Please install it:\n\n"
            "  conda install -c conda-forge umap-learn\n\n"
        )
        raise RuntimeError(msg) from ex

    return UMAP(*args, **kwargs)


def ensure_list(x):
    """
    If ``x`` is already a list, return it unchanged.
    If ``x`` is Series or ndarray, convert to plain list.
    If ``x`` is ``None``, return an empty list ``[]``.
    Otherwise, wrap it in a list.
    """
    if x is None:
        return []

    if isinstance(x, (np.ndarray, pd.Series)):
        return x.tolist()

    if isinstance(x, Collection) and not isinstance(x, str):
        # Note:
        #   This is a convenient way to handle all of these cases:
        #   np.array([1, 2, 3]) -> [1, 2, 3]
        #   [1, 2, 3] -> [1, 2, 3]
        #   [np.int64(1), np.int64(2), np.int64(3)] -> [1, 2, 3]
        return np.asarray(x).tolist()
    else:
        return [x]


def ensure_list_args(argnames):
    """
    Returns a decorator.
    For the given argument names, the decorator converts the
    arguments into iterables via ``ensure_list()``.
    """
    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            callargs = inspect.getcallargs(f, *args, **kwargs)
            for name in argnames:
                callargs[name] = ensure_list(callargs[name])
            return f(**callargs)

        wrapper.__signature__ = inspect.signature(f)
        return wrapper

    return decorator


def ensure_list_attrs(attributes):
    """
    Returns a *class* decorator.
    For the given attribute names, the decorator adds "private"
    attributes (e.g. bodyId -> _bodyId) and declares getter/setter properties.
    The setter property converts the new value to a list before storing
    it in the private attribute.

    Classes which require their members to be a true list can allow users to
    set attributes as np.array.
    """
    def decorator(cls):
        for attr in attributes:
            private_attr = f"_{attr}"

            def getter(self, private_attr=private_attr):
                return getattr(self, private_attr)

            def setter(self, value, private_attr=private_attr):
                value = ensure_list(value)
                setattr(self, private_attr, value)

            setattr(cls, attr, property(getter, setter))

        return cls
    return decorator


@ensure_list_args(['properties'])
def merge_neuron_properties(neuron_df, conn_df, properties=['type', 'instance']):
    """
    Merge neuron properties to a connection table.

    Given a table of neuron properties and a connection table, append
    ``_pre`` and ``_post`` columns to the connection table for each of
    the given properties via the appropriate merge operations.

    Args:
        neuron_df:
            DataFrame with columns for 'bodyId' and any properties you want to merge

        conn_df:
            DataFrame with columns ``bodyId_pre`` and ``bodyId_post``

        properties:
            Column names from ``neuron_df`` to merge onto ``conn_df``.

    Returns:
        Updated ``conn_df`` with new columns.

    Example:

        .. code-block:: ipython

            In [1]: from neuprint import fetch_adjacencies, NeuronCriteria as NC, merge_neuron_properties
               ...: neuron_df, conn_df = fetch_adjacencies(rois='PB', min_roi_weight=120)
               ...: print(conn_df)
               bodyId_pre  bodyId_post roi  weight
            0   880875736   1631450739  PB     123
            1   880880259    849421763  PB     141
            2   910442723    849421763  PB     139
            3   910783961   5813070465  PB     184
            4   911129204    724280817  PB     127
            5   911134009    849421763  PB     125
            6   911565419   5813070465  PB     141
            7   911911004   1062526223  PB     125
            8   911919044    973566036  PB     122
            9  5813080838    974239375  PB     136

            In [2]: merge_neuron_properties(neuron_df, conn_df, 'type')
            Out[2]:
               bodyId_pre  bodyId_post roi  weight  type_pre    type_post
            0   880875736   1631450739  PB     123  Delta7_a  PEN_b(PEN2)
            1   880880259    849421763  PB     141  Delta7_a  PEN_b(PEN2)
            2   910442723    849421763  PB     139  Delta7_a  PEN_b(PEN2)
            3   910783961   5813070465  PB     184  Delta7_a  PEN_b(PEN2)
            4   911129204    724280817  PB     127  Delta7_a  PEN_b(PEN2)
            5   911134009    849421763  PB     125  Delta7_a  PEN_b(PEN2)
            6   911565419   5813070465  PB     141  Delta7_a  PEN_b(PEN2)
            7   911911004   1062526223  PB     125  Delta7_b  PEN_b(PEN2)
            8   911919044    973566036  PB     122  Delta7_a  PEN_b(PEN2)
            9  5813080838    974239375  PB     136       EPG          PEG
    """
    neuron_df = neuron_df[['bodyId', *properties]]

    newcols  = [f'{prop}_pre'  for prop in properties]
    newcols += [f'{prop}_post' for prop in properties]
    conn_df = conn_df.drop(columns=newcols, errors='ignore')

    conn_df = conn_df.merge(neuron_df, 'left', left_on='bodyId_pre', right_on='bodyId')
    del conn_df['bodyId']

    conn_df = conn_df.merge(neuron_df, 'left', left_on='bodyId_post', right_on='bodyId',
                            suffixes=['_pre', '_post'])
    del conn_df['bodyId']

    return conn_df


def connection_table_to_matrix(conn_df, group_cols='bodyId', weight_col='weight', sort_by=None, make_square=False):
    """
    Given a weighted connection table, produce a weighted adjacency matrix.

    Args:
        conn_df:
            A DataFrame with columns for pre- and post- identifiers
            (e.g. bodyId, type or instance), and a column for the
            weight of the connection.

        group_cols:
            Which two columns to use as the row index and column index
            of the returned matrix, respetively.
            Or give a single string (e.g. ``"body"``, in which case the
            two column names are chosen by appending the suffixes
            ``_pre`` and ``_post`` to your string.

            If a pair of pre/post values occurs more than once in the
            connection table, all of its weights will be summed in the
            output matrix.

        weight_col:
            Which column holds the connection weight, to be aggregated for each unique pre/post pair.

        sort_by:
            How to sort the rows and columns of the result.
            Can be two strings, e.g. ``("type_pre", "type_post")``,
            or a single string, e.g. ``"type"`` in which case the suffixes are assumed.

        make_square:
            If True, insert rows and columns to ensure that the same IDs exist in the rows and columns.
            Inserted entries will have value 0.0

    Returns:
        DataFrame, shape NxM, where N is the number of unique values in
        the 'pre' group column, and M is the number of unique values in
        the 'post' group column.

    Example:

        .. code-block:: ipython

            In [1]: from neuprint import fetch_simple_connections, NeuronCriteria as NC
               ...: kc_criteria = NC(type='KC.*')
               ...: conn_df = fetch_simple_connections(kc_criteria, kc_criteria)
            In [1]: conn_df.head()
            Out[1]:
               bodyId_pre  bodyId_post  weight type_pre type_post instance_pre instance_post                                       conn_roiInfo
            0  1224137495   5813032771      29      KCg       KCg          KCg    KCg(super)  {'MB(R)': {'pre': 26, 'post': 26}, 'gL(R)': {'...
            1  1172713521   5813067826      27      KCg       KCg   KCg(super)         KCg-d  {'MB(R)': {'pre': 26, 'post': 26}, 'PED(R)': {...
            2   517858947   5813032943      26   KCab-p    KCab-p       KCab-p        KCab-p  {'MB(R)': {'pre': 25, 'post': 25}, 'PED(R)': {...
            3   642680826   5812980940      25   KCab-p    KCab-p       KCab-p        KCab-p  {'MB(R)': {'pre': 25, 'post': 25}, 'PED(R)': {...
            4  5813067826   1172713521      24      KCg       KCg        KCg-d    KCg(super)  {'MB(R)': {'pre': 23, 'post': 23}, 'gL(R)': {'...

            In [2]: from neuprint.utils import connection_table_to_matrix
               ...: connection_table_to_matrix(conn_df, 'type')
            Out[2]:
            type_post   KC  KCa'b'  KCab-p  KCab-sc     KCg
            type_pre
            KC           3     139       6        5     365
            KCa'b'     154  102337     245      997    1977
            KCab-p       7     310   17899     3029     127
            KCab-sc      4    2591    3975   247038    3419
            KCg        380    1969      79     1526  250351
    """
    if isinstance(group_cols, str):
        group_cols = (f"{group_cols}_pre", f"{group_cols}_post")

    assert len(group_cols) == 2, \
        "Please provide two group_cols (e.g. 'bodyId_pre', 'bodyId_post')"

    assert group_cols[0] in conn_df, \
        f"Column missing: {group_cols[0]}"

    assert group_cols[1] in conn_df, \
        f"Column missing: {group_cols[1]}"

    assert weight_col in conn_df, \
        f"Column missing: {weight_col}"

    col_pre, col_post = group_cols
    dtype = conn_df[weight_col].dtype

    agg_weights_df = conn_df.groupby([col_pre, col_post], sort=False)[weight_col].sum().reset_index()
    matrix = agg_weights_df.pivot(index=col_pre, columns=col_post, values=weight_col)
    matrix = matrix.fillna(0).astype(dtype)

    if sort_by:
        if isinstance(sort_by, str):
            sort_by = (f"{sort_by}_pre", f"{sort_by}_post")

        assert len(sort_by) == 2, \
            "Please provide two sort_by column names (e.g. 'type_pre', 'type_post')"

        pre_order = conn_df.sort_values(sort_by[0])[col_pre].unique()
        post_order = conn_df.sort_values(sort_by[1])[col_post].unique()
        matrix = matrix.reindex(index=pre_order, columns=post_order)
    else:
        # No sort: Keep the order as close to the input order as possible.
        pre_order = conn_df[col_pre].unique()
        post_order = conn_df[col_post].unique()
        matrix = matrix.reindex(index=pre_order, columns=post_order)

    if make_square:
        matrix, _ = matrix.align(matrix.T)
        matrix = matrix.fillna(0.0).astype(matrix.dtypes)

        matrix = matrix.rename_axis(col_pre, axis=0).rename_axis(col_post, axis=1)
        matrix = matrix.loc[
            sorted(matrix.index, key=lambda s: s if s else ""),
            sorted(matrix.columns, key=lambda s: s if s else "")
        ]

    return matrix


def iter_batches(it, batch_size):
    """
    Iterator.

    Consume the given iterator/iterable in batches and
    yield each batch as a list of items.

    The last batch might be smaller than the others,
    if there aren't enough items to fill it.

    If the given iterator supports the __len__ method,
    the returned batch iterator will, too.
    """
    if hasattr(it, '__len__'):
        return _iter_batches_with_len(it, batch_size)
    else:
        return _iter_batches(it, batch_size)


class _iter_batches:
    def __init__(self, it, batch_size):
        self.base_iterator = it
        self.batch_size = batch_size


    def __iter__(self):
        return self._iter_batches(self.base_iterator, self.batch_size)


    def _iter_batches(self, it, batch_size):
        if isinstance(it, (pd.DataFrame, pd.Series)):
            for batch_start in range(0, len(it), batch_size):
                yield it.iloc[batch_start:batch_start+batch_size]
            return
        elif isinstance(it, (list, np.ndarray)):
            for batch_start in range(0, len(it), batch_size):
                yield it[batch_start:batch_start+batch_size]
            return
        else:
            if not isinstance(it, Iterator):
                assert isinstance(it, Iterable)
                it = iter(it)

            while True:
                batch = []
                try:
                    for _ in range(batch_size):
                        batch.append(next(it))
                except StopIteration:
                    return
                finally:
                    if batch:
                        yield batch


class _iter_batches_with_len(_iter_batches):
    def __len__(self):
        return int(np.ceil(len(self.base_iterator) / self.batch_size))


def compile_columns(client, user_columns=[], core_columns=[], core_columns_only=False):
    """
    Compile list of columns from user input and available :Neuron keys (excluding ROIs).

    Args:
        client:
            neu.Client to collect columns for.
        user_columns:
            List of user-defined columns (optional). The list will be returned in
            input order, with non-existing columns dropped.
        core_columns:
            List of core columns (optional). If provided, core columns will be
            appear at the beginning of the list, and non-existing columns will be
            dropped.
        core_columns_only: default=False
            If True and user_columns is empty, only return columns that are in
            the core_columns list (which should not be empty).

    Returns:
        columns:
            List of key names, with core columns first, followed by other
            columns in sorted order.
    """
    # Fetch existing keys. This call is cached.
    keys = client.fetch_neuron_keys()

    # Drop ROIs
    keys = [k for k in keys if k not in client.all_rois]

    if user_columns:
        # user columns only, in the order they were given, if they exist
        columns = [k for k in user_columns if k in keys]
    else:
        # core columns go first, if they exist
        columns = [k for k in core_columns if k in keys]
        if not core_columns_only:
            # add in the other keys, in sorted order
            columns += [k for k in sorted(keys) if k not in columns]

    return columns

def available_datasets(server, token=None):
    """
    Get a list of available datasets for a specified server.
    Args:
        server: URL of neuprintHttp server
        token: neuPrint token. If null, will use
               ``NEUPRINT_APPLICATION_CREDENTIALS`` environment variable.
               Your token can be retrieved by clicking on your account in
               the NeuPrint web interface.
    Returns:
        List of available datasets
    """
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
    while server.endswith('/'):
        server = server[:-1]
    # Request
    with Session() as session:
        session.headers.update({'Authorization': f'Bearer {token}'})
        response = session.get(f"{server}/api/dbmeta/datasets")
        response.raise_for_status()
    return list(response.json())
