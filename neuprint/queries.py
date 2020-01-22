import os
import pandas as pd
from tqdm import trange

from .utils import make_iterable, parse_properties
from .client import inject_client

try:
    # ujson is faster than Python's builtin json module;
    # use it if the user happens to have it installed.
    import ujson as json
except ImportError:
    import json


@inject_client
def fetch_custom(cypher, dataset="", format='pandas', *, client=None):
    """
    Alternative form of Client.fetch_custom(), as a free function.
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
            or return the server's raw JSON response as a Python dict.

        client:
            If not provided, the global default ``Client`` will be used.
    
    Returns:
        Either json or DataFrame, depending on ``format``.
    """
    return client.fetch_custom(cypher, dataset, format)


@inject_client
def fetch_traced_adjacencies(export_dir=None, batch_size=200, *, client=None):
    """
    Fetch the adjacency table for all non-cropped traced neurons, broken down by ROI.
    Synapses which do not fall on any ROI will be listed as having ROI 'None'.
    
    
    Args:
        export_dir:
            Optional. Export CSV files for the neuron table,
            connection table (total weight), and connection table (per ROI).
            
        batch_size:
            For optimal performance, connections will be fetched in batches.
            This parameter specifies the batch size.
    
    Returns:
        Two DataFrames, ``(traced_neurons_df, roi_conn_df)``, containing the
        table of neuron IDs and the per-ROI connection table, respectively.

    Note:
        On the hemibrain dataset, this function takes a few minutes to run,
        and the results results are somewhat large (~300 MB).
    
    Example:
        
        .. code-block:: ipython
        
            In [1]: neurons_df, roi_conn_df = fetch_traced_adjacencies('exported-connections')

            In [2]: roi_conn_df.head()
            Out[2]:
                   bodyId_pre  bodyId_post        roi  weight
            0      5813009352    516098538     SNP(R)       2
            1      5813009352    516098538     SLP(R)       2
            2       326119769    516098538     SNP(R)       1
            3       326119769    516098538     SLP(R)       1
            4       915960391    202916528         FB       1

            In [3]: # Obtain total weights (instead of per-connection-per-ROI weights)
               ...: conn_groups = roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)
               ...: total_conn_df = conn_groups['weight'].sum()
               ...: total_conn_df.head()
            Out[3]:
               bodyId_pre  bodyId_post  weight
            0   202916528    203253253       2
            1   202916528    203257652       2
            2   202916528    203598557       2
            3   202916528    234292899       4
            4   202916528    264986706       2        
    """
    ##
    ## TODO: Options to specify non-cropped, etc.
    ##
    
    # Fetch the list of primary ROIs
    q = """
        MATCH (m:Meta)
        RETURN m.primaryRois as rois
    """
    primary_rois = client.fetch_custom(q)['rois'].iloc[0]
    
    # Fetch the list of traced, non-cropped Neurons
    q = """\
        MATCH (n:Neuron)
        WHERE n.status = "Traced" AND (not n.cropped)
        RETURN n.bodyId as bodyId, n.instance as instance, n.type as type
    """
    traced_neurons_df = client.fetch_custom(q)
    
    # Fetch connections in batches
    conn_tables = []
    for start in trange(0, len(traced_neurons_df), batch_size):
        stop = start + batch_size
        batch_neurons = traced_neurons_df['bodyId'].iloc[start:stop].tolist()
        q = f"""\
            MATCH (n:Neuron) - [e:ConnectsTo] -> (m:Neuron)
            WHERE n.bodyId in {batch_neurons} AND m.status = "Traced" AND (not m.cropped)
            RETURN n.bodyId as bodyId_pre, m.bodyId as bodyId_post, e.weight as weight, e.roiInfo as roiInfo
        """
        conn_tables.append( client.fetch_custom(q) )
    
    # Combine batches
    connections_df = pd.concat(conn_tables, ignore_index=True)
    
    # Parse roiInfo json
    connections_df['roiInfo'] = connections_df['roiInfo'].apply(json.loads)

    # Extract per-ROI counts from the roiInfo column
    # to construct one big table of per-ROI counts
    roi_connections = []
    for row in connections_df.itertuples(index=False):
        # We use the 'post' count as the weight (ignore pre)
        roi_connections += [(row.bodyId_pre, row.bodyId_post, roi, weights.get('post', 0))
                            for roi, weights in row.roiInfo.items()]
    
    roi_conn_df = pd.DataFrame(roi_connections,
                               columns=['bodyId_pre', 'bodyId_post', 'roi', 'weight'])
    
    # Filter out non-primary ROIs
    roi_conn_df = roi_conn_df.query('roi in @primary_rois or roi == "None"')
    
    # Export to CSV
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

        # Export Nodes
        p = f"{export_dir}/traced-neurons.csv"
        traced_neurons_df.to_csv(p, index=False, header=True)
        
        # Export Edges (per ROI)
        p = f"{export_dir}/traced-roi-connections.csv"
        roi_conn_df.to_csv(p, index=False, header=True)

        # Export Edges (total weight)
        p = f"{export_dir}/traced-connections.csv"
        conn_groups = roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)
        total_conn_df = conn_groups['weight'].sum()
        total_conn_df.to_csv(p, index=False, header=True)

    return traced_neurons_df, roi_conn_df
