import pandas as pd
import numpy as np

from .utils import make_iterable, parse_properties
from .client import inject_client


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
def custom_search(x, props=['bodyId', 'name'], logic='AND', dataset='hemibrain',
                  datatype='Neuron', *, client=None):
    """ Find neurons by neo4j WHERE query.

    Parameters
    ----------
    x :         str | iterable, optional
                Must be valid neo4j ``WHERE`` query e.g. ``'status="Anchor"'``
                or ``"bodyId IN [123456,56688]"``
    props :     iterable, optional
                Neuron properties to return.
    logic :     "AND" | "OR", optional
                Logic to apply when multiple search parameters are given.
    dataset :   str, optional
                Which dataset to query. See ``neuprint.Client.fetch_datasets``
                for available datasets.
    datatype :  str, optional
                Data type to search for. Depends on dataset. For
                ``dataset='hemibrain'`` options are "Neuron" and "Segment".
                The former is limited to bodies with either >=2 pre-, >= 10
                postsynapses, name, soma or status.
    client :    neuprint.Client, optional
                If ``None`` will try using global client.

    Returns
    -------
    pandas.DataFrame
    """

    x = make_iterable(x)

    where = ' {} '.format(logic).join(['n.{}'.format(s) for s in x])
    ret = parse_properties(props, 'n')

    cypher = """
             MATCH (n :`{datatype}`)
             WHERE {where}
             RETURN {ret}
             """.format(dataset=dataset, datatype=datatype, where=where,
                        ret=ret)

    return client.fetch_custom(cypher, dataset=dataset)


@inject_client
def fetch_neurons_in_roi(roi, dataset='hemibrain', datatype='Neuron',
                         logic='AND', add_props=None, *, client=None):
    """ Fetch all neurons within given ROI.

    Parameters
    ----------
    roi :       str | iterable
                ROI(s) (e.g. "LH") to query. See ``neuprint.Client.fetch_datasets``
                for available ROIs. Use a tilde (~) prefix to exclude neurons that
                have arbors in a given ROI.
    dataset :   str, optional
                Which dataset to query. See ``neuprint.Client.fetch_datasets``
                for available datasets.
    datatype :  str, optional
                Data type to search for. Depends on dataset. For
                ``dataset='hemibrain'`` options are "Neuron" and "Segment".
                The former is limited to bodies with either >=2 pre-, >= 10
                postsynapses, name, soma or status.
    logic :     "AND" | "OR", optional
                Logic to apply when multiple ROIs are queried.
    add_props : iterable, optional
                Additional neuron properties to be returned.
    client :    neuprint.Client, optional
                If ``None`` will try using global client.

    Returns
    -------
    pandas.DataFrame
    """

    roi = make_iterable(roi)

    # Parse ROI
    conditions = []
    for r in roi:
        if r.startswith('~'):
            conditions.append('NOT exists(n.`{}`)'.format(r[1:]))
        else:
            conditions.append('n.`{}`=true'.format(r))

    # Now remove tildes
    roi = [r.replace('~', '') for r in roi]

    cypher = """
             MATCH (n :`{dataset}-{datatype}`)
             WHERE {where} WITH n AS n, apoc.convert.fromJsonMap(n.roiInfo) AS roiInfo
             RETURN n.bodyId AS bodyId, n.size AS size, n.status AS status,
                    n.pre AS pre, n.post AS post, {roiPre}, {roiPost}
             """.format(dataset=dataset, datatype=datatype,
                        roiPre=', '.join(['roiInfo.{0}.pre as pre_{0}'.format(r) for r in roi]),
                        roiPost=', '.join(['roiInfo.{0}.post as post_{0}'.format(r) for r in roi]),
                        where=' {} '.format(logic).join(conditions))

    if add_props:
        add_props = add_props if isinstance(add_props, list) else list(add_props)
        cypher += ','
        cypher += ','.join(['n.{0} AS {0}'.format(p) for p in add_props])

    return client.fetch_custom(cypher)


@inject_client
def find_neurons(x, dataset='hemibrain', datatype='Neuron', add_props=None, *, client=None):
    """ Find neurons by name or body ID.

    Parameters
    ----------
    x :         str | int | list-like | pandas.DataFrame
                Search string. Can be body ID(s), neuron name or wildcard/regex
                names (e.g. "MBON.*"). Body IDs can also be provided as
                list-like or DataFrame with "bodyId" column.
    dataset :   str, optional
                Which dataset to query. See ``neuprint.Client.fetch_datasets``
                for available datasets.
    datatype :  str, optional
                Data type to search for. Depends on dataset. For
                ``dataset='hemibrain'`` options are "Neuron" and "Segment".
                The former is limited to bodies with either >=2 pre-, >= 10
                postsynapses, name, soma or status.
    add_props : iterable, optional
                Additional neuron properties to be returned.
    client :    neuprint.Client, optional
                If ``None`` will try using global client.

    Returns
    -------
    pandas.DataFrame
    """

    if isinstance(x, pd.DataFrame):
        if 'bodyId' in x.columns:
            x = x['bodyId'].values
        else:
            raise ValueError('DataFrame must have "bodyId" column.')

    if isinstance(x, str):
        if x.isnumeric():
            where = 'bodyId={}'.format(x)
        else:
            where = 'name=~"{}"'.format(x)
    elif isinstance(x, (list, tuple, np.ndarray)):
        if all([isinstance(s, str) for s in x]):
            if all([s.isnumeric() for s in x]):
                where = 'bodyId IN {}'.format(list(np.array(x).astype(int)))
            else:
                raise ValueError('List can only be numeric body IDs')
        elif all([isinstance(s, (int, np.int64, np.int32)) for s in x]):
            where = 'bodyId IN {}'.format(list(x))
        else:
            raise ValueError('List can only be numeric body IDs')
    elif isinstance(x, (int, np.int64, np.int32)):
        where = 'bodyId={}'.format(x)
    else:
        raise ValueError('Unable to process data of type "{}"'.format(type(x)))

    props = ['bodyId', 'name', 'size', 'status', 'pre', 'post']

    if add_props:
        props += add_props if isinstance(add_props, list) else list(add_props)
        props = list(set(props))

    return custom_search(where, props=props, dataset=dataset,
                         datatype=datatype, client=client)


@inject_client
def fetch_connectivity(x, dataset='hemibrain', datatype='Neuron', add_props=None, *, client=None):
    """ Fetch connectivity table for given neuron

    Parameters
    ----------
    x :         str | int | iterable
                Neuron filter. Can be body ID, neuron name or wildcard names
                (e.g. "MBON.*"). Accepts regex. Body IDs can be given as
                list.
    dataset :   str, optional
                Which dataset to query. See ``neuprint.Client.fetch_datasets``
                for available datasets.
    datatype :  str, optional
                Data type to search for. Depends on dataset. For
                ``dataset='hemibrain'`` options are "Neuron" and "Segment".
                The former is limited to bodies with either >=2 pre-, >= 10
                postsynapses, name, soma or status.
    add_props : iterable, optional
                Additional neuron properties to be returned.
    client :    neuprint.Client, optional
                If ``None`` will try using global client.

    Returns
    -------
    pandas.DataFrame
    """

    if isinstance(x, pd.DataFrame):
        if 'bodyId' in x.columns:
            x = x['bodyId'].values
        else:
            raise ValueError('DataFrame must have "bodyId" column.')

    pre = ''

    if isinstance(x, str):
        if x.isnumeric():
            where = 'bodyId={}'.format(x)
        else:
            where = 'name=~"{}"'.format(x)
    elif isinstance(x, (np.ndarray, list, tuple)):
        where = 'bodyId=bid'
        pre = 'WITH {} AS bodyIds UNWIND bodyIds AS bid'.format(list(x))
    else:
        where = 'bodyId={}'.format(x)

    ret = ['m.name AS name1', 'n.name AS name2', 'e.weight AS weight',
           'n.bodyId AS body2', 'id(m) AS id1', 'id(n) AS id2',
           'id(startNode(e)) AS pre_id', 'm.bodyId AS body1',
           'e.weightHP AS WeightHP']

    if add_props:
        ret += ['n.{} as {}'.format(p, p) for p in add_props]

    cypher = """
             {pre}
             MATCH (m:`{dataset}-{datatype}`)-[e:ConnectsTo]-(n)
             WHERE m.{where}
             RETURN {ret}
             """.format(dataset=dataset, datatype=datatype, where=where,
                        ret=', '.join(ret), pre=pre)

    # Fetch data
    data = client.fetch_custom(cypher)

    # Try converting to numeric
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='ignore')

    # Add relation
    data['relation'] = 'upstream'
    data.loc[data.pre_id == data.id1, 'relation'] = 'downstream'

    # Now we need to bring this in the right format
    # Pivot such that each row is a connected neuron
    to_keep = ['name2', 'body2', 'relation', 'weight']
    if add_props:
        to_keep += ['{}'.format(p) for p in add_props]

    p = data.pivot(columns='body1', values=to_keep)

    # Try converting to numeric
    for c in p.columns:
        p[c] = pd.to_numeric(p[c], errors='ignore')

    # Reconstruct DataFrame
    df = pd.DataFrame()
    # Combine non-weight columns
    for c in [c for c in p.columns.levels[0] if c != 'weight']:
        df[c] = p[(c, p.columns.levels[1][0])]
        for l in [l for l in p.columns.levels[1] if l]:
            df[c].fillna(p[(c, l)], inplace=True)
    # Add weight column and fillna
    for l in [l for l in p.columns.levels[1] if l]:
        df[l] = p[('weight', l)]

    # Rename some columns
    to_replace = {'body2': 'bodyId', 'name2': 'name'}
    df.columns = [to_replace.get(c, c) for c in df.columns]

    # Make bodyId column integer
    df['bodyId'] = df.bodyId.astype(int)

    # Neurons will still show up multiple times -> group and keep the first
    # non-NaN value
    return df.groupby(['bodyId', 'relation']).first().reset_index(drop=False).fillna(0)


@inject_client
def fetch_connectivity_in_roi(roi, source=None, target=None, logic='AND',
                              dataset='hemibrain', datatype='Neuron',
                              add_props=None, *, client=None):
    """Fetch connectivity within ROI between given neuron(s).

    Parameters
    ----------
    roi :       str | list
                ROI(s) to filter for. Prefix the ROI with a tilde (~) to return
                everything OUTSIDE the ROI.
    source :    str | int | iterable | None, optional
                Source neurons. Can be body ID, neuron name or wildcard names
                (e.g. "MBON.*"). Accepts regex. Body IDs can be given as
                list. If ``None`` will get all inputs to ``target``.
    target :    str | int | iterable | None
                Target neurons. If ``None`` will get all outputs of ``sources``.
    logic :     "AND" | "OR", optional
                Logic to apply when multiple ROIs are queried.
    dataset :   str, optional
                Which dataset to query. See ``neuprint.Client.fetch_datasets``
                for available datasets.
    datatype :  str, optional
                Data type to search for. Depends on dataset. For
                ``dataset='hemibrain'`` options are "Neuron" and "Segment".
                The former is limited to bodies with either >=2 pre-, >= 10
                postsynapses, name, soma or status.
    add_props : iterable, optional
                Additional neuron properties to be returned.
    client :    neuprint.Client, optional
                If ``None`` will try using global client.

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    Find all downstream targets outside of calyx

    >>> ds = neuprint.fetch_connectivity('~CA', source=123456)

    Produce CATMAID style connectivity table

    >>> data = neuprint.fetch_connectivity_in_roi('ROI', source=123456)
    >>> cn = data.pivot(index='source', columns='target', values='synapses').T
    >>> cn.fillna(0, inplace=True)
    >>> cn['total'] = cn.sum(axis=1)
    >>> cn.sort_values('total', inplace=True, ascending=False)
    >>> names = neuprint.find_neurons(cn.index.values)
    >>> names = names.set_index('bodyId').to_dict()
    >>> cn['name'] = cn.index.map(lambda x: names['name'].get(x, None))
    >>> cn['size'] = cn.index.map(lambda x: names['size'].get(x, None))
    >>> cn['status'] = cn.index.map(lambda x: names['status'].get(x, None))

    """

    if isinstance(source, type(None)) and isinstance(target, type(None)):
        raise ValueError('source and target must not both be "None"')

    if isinstance(source, pd.DataFrame):
        if 'bodyId' in source.columns:
            source = source['bodyId'].values
        else:
            raise ValueError('DataFrame must have "bodyId" column.')

    if isinstance(target, pd.DataFrame):
        if 'bodyId' in target.columns:
            target = target['bodyId'].values
        else:
            raise ValueError('DataFrame must have "bodyId" column.')

    roi = make_iterable(roi)

    # Parse ROI restrictions
    conditions = []
    for r in roi:
        if r.startswith('~'):
            conditions.append('NOT exists(s.`{}`)'.format(r[1:]))
        else:
            conditions.append('exists(s.`{}`)'.format(r))

    where = '({})'.format(' {} '.format(logic).join(conditions))
    where += ' AND (s.type="post")'

    pre_with = ''
    pre_unwind = ''

    if not isinstance(source, type(None)):
        if isinstance(source, str):
            if source.isnumeric():
                where += ' AND a.bodyId={}'.format(source)
            else:
                where += ' AND a.name=~"{}"'.format(source)
        elif isinstance(source, (np.ndarray, list, tuple)):
            where += ' AND a.bodyId=sid'
            pre_with = 'WITH {} AS sourceIds'.format(list(np.array(source).astype(int)))
            pre_unwind = 'UNWIND sourceIds AS sid'
        else:
            where += ' AND a.bodyId={}'.format(source)

    if not isinstance(target, type(None)):
        if isinstance(target, str):
            if target.isnumeric():
                where += ' AND b.bodyId={}'.format(target)
            else:
                where += ' AND b.name=~"{}"'.format(target)
        elif isinstance(target, (np.ndarray, list, tuple)):
            where += ' AND b.bodyId=tid'
            if not pre_with:
                pre_with = 'WITH {} AS targetIds'.format(list(np.array(target).astype(int)))
                pre_unwind += 'UNWIND targetIds AS tid'
            else:
                pre_with += ', {} AS targetIds'.format(list(np.array(target).astype(int)))
                pre_unwind += '\nUNWIND targetIds AS tid'
        else:
            where += ' AND b.bodyId={}'.format(target)

    ret = ['a.bodyId AS source', 'b.bodyId AS target', 'count(*) AS synapses']

    if add_props:
        ret += ['a.{} AS source_{}'.format(p, p) for p in add_props]
        ret += ['b.{} AS target_{}'.format(p, p) for p in add_props]

    cypher = """
             {pre_with} {pre_unwind}
             MATCH (a:`{dataset}-{datatype}`)<-[:From]-(c:ConnectionSet)-[:To]->(b:`{dataset}-{datatype}`), (c)-[:Contains]->(s:Synapse)
             WHERE {where}
             RETURN {ret}
             """.format(dataset=dataset, datatype=datatype, where=where,
                        pre_with=pre_with, pre_unwind=pre_unwind,
                        ret=', '.join(ret))

    # Fetch data
    data = client.fetch_custom(cypher)

    return data.sort_values('synapses', ascending=False).reset_index(drop=True)


@inject_client
def fetch_edges(source, target=None, roi=None, dataset='hemibrain',
                datatype='Neuron', add_props=None, *, client=None):
    """Fetch edges between given neuron(s).

    Parameters
    ----------
    source :    str | int | iterable | None, optional
                Source neurons. Can be body ID, neuron name or wildcard names
                (e.g. "MBON.*"). Accepts regex. Body IDs can be given as
                list. If ``None`` will get all inputs to ``target``.
    target :    str | int | iterable | None
                Target neurons. If ``None`` will include all targets of
                ``source``.
    roi :       str
                ROI(s) to restrict connectivity to. Use tilde (~) to exclude
                connections within this ROI.
    dataset :   str, optional
                Which dataset to query. See ``neuprint.Client.fetch_datasets``
                for available datasets.
    datatype :  str, optional
                Data type to search for. Depends on dataset. For
                ``dataset='hemibrain'`` options are "Neuron" and "Segment".
                The former is limited to bodies with either >=2 pre-, >= 10
                postsynapses, name, soma or status.
    add_props : iterable, optional
                Additional neuron properties to be returned.
    client :    neuprint.Client, optional
                If ``None`` will try using global client.

    Returns
    -------
    pandas.DataFrame

    """
    if isinstance(source, pd.DataFrame):
        if 'bodyId' in source.columns:
            source = source['bodyId'].values
        else:
            raise ValueError('DataFrame must have "bodyId" column.')

    if isinstance(target, pd.DataFrame):
        if 'bodyId' in target.columns:
            target = target['bodyId'].values
        else:
            raise ValueError('DataFrame must have "bodyId" column.')

    if isinstance(source, type(None)) and isinstance(target, type(None)):
        raise ValueError('source and target must not both be "None"')

    where = ['(s.type="post")'.format(roi)]
    if not isinstance(roi, type(None)):
        if not isinstance(roi, str):
            raise TypeError('Expected ROI as str, got "{}"'.format(type(roi)))
        if roi.startswith('~'):
            where.append('NOT (exists(s.`{}`))'.format(roi[1:]))
        else:
            where.append('(exists(s.`{}`))'.format(roi))

    pre_with = []
    pre_unwind = []

    if not isinstance(source, type(None)):
        if isinstance(source, str):
            if source.isnumeric():
                where.append('a.bodyId={}'.format(source))
            else:
                where.append('a.name=~"{}"'.format(source))
        elif isinstance(source, (np.ndarray, list, tuple)):
            where.append('a.bodyId=sid')
            pre_with.append('{} AS sourceIds'.format(list(np.array(source).astype(int))))
            pre_unwind.append('sourceIds AS sid')
        else:
            where.append('a.bodyId={}'.format(source))

    if not isinstance(target, type(None)):
        if isinstance(target, str):
            if target.isnumeric():
                where.append('b.bodyId={}'.format(target))
            else:
                where.append('b.name=~"{}"'.format(target))
        elif isinstance(target, (np.ndarray, list, tuple)):
            pre_with.append('{} AS targetIds'.format(list(np.array(target).astype(int))))
            if not pre_with:
                # Only unwind targets if we aren't already unwinding sources
                pre_unwind.append('targetIds AS tid')
                where.append('b.bodyId=tid')
            else:
                where.append('b.bodyId IN targetIds')
        else:
            where.append('b.bodyId={}'.format(target))

    ret = ['a.bodyId AS source', 'b.bodyId AS target', 'count(*) AS synapses']

    if add_props:
        ret += ['a.{} AS source_{}'.format(p, p) for p in add_props]
        ret += ['b.{} AS target_{}'.format(p, p) for p in add_props]

    cypher = """
             WITH {pre_with}
             UNWIND {pre_unwind}
             MATCH (a:`{dataset}-{datatype}`)<-[:From]-(c:ConnectionSet)-[:To]->(b:`{dataset}-{datatype}`), (c)-[:Contains]->(s:Synapse)
             WHERE {where}
             RETURN {ret}
             """.format(dataset=dataset, datatype=datatype,
                        pre_with=', '.join(pre_with),
                        pre_unwind=', '.join(pre_unwind),
                        where=' AND '.join(where),
                        ret=', '.join(ret))

    # Fetch data
    data = client.fetch_custom(cypher)

    return data.sort_values('synapses', ascending=False).reset_index(drop=True)


@inject_client
def fetch_synapses(x, dataset='hemibrain', datatype='Neuron', *, client=None):
    """ Fetch synapses for given body ID(s)

    Parameters
    ----------
    x :             str | int | list-like | pandas.DataFrame
                    Search string. Can be body ID(s), neuron name or
                    wildcard/regex names (e.g. "MBON.*"). Body IDs can also be
                    provided as list-like or DataFrame with "bodyId" column.
    dataset :       str, optional
                    Which dataset to query. See ``neuprint.Client.fetch_datasets``
                    for available datasets.
    datatype :      str, optional
                    Data type to search for. Depends on dataset. For
                    ``dataset='hemibrain'`` options are "Neuron" and "Segment".
                    The former is limited to bodies with either >=2 pre-, >= 10
                    postsynapses, name, soma or status.
    client :        neuprint.Client, optional
                    If ``None`` will try using global client.

    Returns
    -------
    pandas.DataFrame
    """

    if isinstance(x, pd.DataFrame):
        if 'bodyId' in x.columns:
            x = x['bodyId'].values
        else:
            raise ValueError('DataFrame must have "bodyId" column.')

    pre = ''

    if isinstance(x, str):
        if x.isnumeric():
            where = 'bodyId={}'.format(x)
        else:
            where = 'name=~"{}"'.format(x)
    elif isinstance(x, (np.ndarray, list, tuple)):
        where = 'bodyId=bid'
        pre = 'WITH {} AS bodyIds UNWIND bodyIds AS bid'.format(list(x))
    else:
        where = 'bodyId={}'.format(x)

    ret = ['n.bodyId as bodyId', 's']

    cypher = """
             {pre}
             MATCH (n:`{dataset}-{datatype}`)-[:Contains]->(ss:SynapseSet),
                   (ss)-[:Contains]->(s:Synapse)
             WHERE n.{where}
             RETURN {ret}
             """.format(dataset=dataset, datatype=datatype, where=where,
                        ret=', '.join(ret), pre=pre)

    # Get data
    r = fetch_custom(cypher, client=client, format='json')

    # Flatten Synapse data
    s = pd.io.json.json_normalize([s[1] for s in r['data']])
    s['bodyId'] = [s[0] for s in r['data']]

    return s.fillna(False)
