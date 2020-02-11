"""
Utility functions for manipulating neuprint-python output.
"""
import sys
import inspect
import functools
from collections.abc import Iterable

#
# Import the notebook-aware version of tqdm if
# we appear to be running within a notebook context.
#
try:
    import ipykernel.iostream
    if isinstance(sys.stdout, ipykernel.iostream.OutStream):
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm, trange
except ImportError:
    from tqdm import tqdm, trange


def make_iterable(x):
    """
    If ``x`` is already a list or array, return it unchanged.
    If ``x`` is ``None``, return an empty list ``[]``.
    Otherwise, wrap it in a list.
    """
    if x is None:
        return []
    if isinstance(x, Iterable) and not isinstance(x, str):
        return x
    else:
        return [x]


def make_args_iterable(argnames):
    """
    Returns a decorator.
    For the given argument names, the decorator converts the
    arguments into iterables via ``make_iterable()``.
    """
    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            callargs = inspect.getcallargs(f, *args, **kwargs)
            for name in argnames:
                callargs[name] = make_iterable(callargs[name])
            return f(**callargs)

        wrapper.__signature__ = inspect.signature(f)
        return wrapper

    return decorator


@make_args_iterable(['properties'])
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
    
            In [1]: from neuprint import fetch_adjacencies, SegmentCriteria as SC, merge_neuron_properties
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


def connection_table_to_matrix(conn_df, group_cols='bodyId', weight_col='weight', sort_by=None):
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
            
    Returns:
        DataFrame, shape NxM, where N is the number of unique values in
        the 'pre' group column, and M is the number of unique values in
        the 'post' group column.
    
    Example:
    
        .. code-block:: ipython
        
            In [1]: from neuprint import fetch_simple_connections, SegmentCriteria as SC  
               ...: kc_criteria = SC(type='KC.*', regex=True)
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

    grouped = conn_df.groupby([col_pre, col_post], as_index=False, sort=False)
    agg_weights_df = grouped[weight_col].sum()
    matrix = agg_weights_df.pivot(col_pre, col_post, weight_col)
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

    return matrix
