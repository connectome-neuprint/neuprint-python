"""
Deprecated functions.

Deprecated convenience functions have been moved to ``neuprint.deprecated``.
They will soon be replaced with alternatives in the main ``neuprint-python`` API,
but they remain available here for users of the previous API.

Note:

    Some of these functions may result in errors, since they were written
    to use an older version of neuprint's data model.
"""
import pandas as pd
import numpy as np

from ..utils import make_iterable
from ..client import inject_client


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

    where = f' {logic} '.join([f'n.{s}' for s in x])
    ret = parse_properties(props, 'n')

    cypher = f"""\
             MATCH (n :`{dataset}_{datatype}`)
             WHERE {where}
             RETURN {ret}
             """

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
            conditions.append(f'NOT exists(n.`{r[1:]}`)')
        else:
            conditions.append(f'n.`{r}`=true')

    # Now remove tildes
    roi = [r.replace('~', '') for r in roi]

    roiPre = ', '.join([f'roiInfo.`{r}`.pre as `pre_{r}`' for r in roi])
    roiPost = ', '.join([f'roiInfo.`{r}`.post as `post_{r}`' for r in roi])
    where = ' {logic} '.join(conditions)

    cypher = f"""\
             MATCH (n :`{dataset}_{datatype}`)
             WHERE {where} WITH n AS n, apoc.convert.fromJsonMap(n.roiInfo) AS roiInfo
             RETURN n.bodyId AS bodyId, n.size AS size, n.status AS status,
                    n.pre AS pre, n.post AS post, {roiPre}, {roiPost}
             """

    if add_props:
        add_props = add_props if isinstance(add_props, list) else list(add_props)
        cypher += ','
        cypher += ','.join([f'n.{p} AS {p}' for p in add_props])

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
            where = f'bodyId={x}'
        else:
            where = f'name=~"{x}"'
    elif isinstance(x, (list, tuple, np.ndarray)):
        if all([isinstance(s, str) for s in x]):
            if all([s.isnumeric() for s in x]):
                body_list = np.array(x).tolist()
                where = f'bodyId IN {body_list}'
            else:
                raise ValueError('List can only be numeric body IDs')
        elif all([isinstance(s, (int, np.int64, np.int32)) for s in x]):
            where = f'bodyId IN {list(x)}'
        else:
            raise ValueError('List can only be numeric body IDs')
    elif isinstance(x, (int, np.int64, np.int32)):
        where = f'bodyId={x}'
    else:
        raise ValueError(f'Unable to process data of type "{type(x)}"')

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
            where = f'bodyId={x}'
        else:
            where = f'name=~"{x}"'
    elif isinstance(x, (np.ndarray, list, tuple)):
        where = 'bodyId=bid'
        pre = f'WITH {list(x)} AS bodyIds UNWIND bodyIds AS bid'
    else:
        where = f'bodyId={x}'

    ret = ['m.name AS name1', 'n.name AS name2', 'e.weight AS weight',
           'n.bodyId AS body2', 'id(m) AS id1', 'id(n) AS id2',
           'id(startNode(e)) AS pre_id', 'm.bodyId AS body1',
           'e.weightHP AS WeightHP']

    if add_props:
        ret += [f'n.{p} as {p}' for p in add_props]

    ret = ', '.join(ret)
    
    cypher = f"""\
             {pre}
             MATCH (m:`{dataset}_{datatype}`)-[e:ConnectsTo]-(n)
             WHERE m.{where}
             RETURN {ret}
             """

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
        to_keep += [f'{p}' for p in add_props]

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
            conditions.append(f'NOT exists(s.`{r[1:]}`)')
        else:
            conditions.append(f'exists(s.`{r}`)')

    conditions = f' {logic} '.join(conditions)
    where = f'({conditions})'
    where += ' AND (s.type="post")'

    pre_with = ''
    pre_unwind = ''

    if not isinstance(source, type(None)):
        if isinstance(source, str):
            if source.isnumeric():
                where += f' AND a.bodyId={source}'
            else:
                where += f' AND a.name=~"{source}"'
        elif isinstance(source, (np.ndarray, list, tuple)):
            source = np.array(source).tolist()
            where += ' AND a.bodyId=sid'
            pre_with = f'WITH {source} AS sourceIds'
            pre_unwind = 'UNWIND sourceIds AS sid'
        else:
            where += f' AND a.bodyId={source}'

    if not isinstance(target, type(None)):
        if isinstance(target, str):
            if target.isnumeric():
                where += f' AND b.bodyId={target}'
            else:
                where += f' AND b.name=~"{target}"'
        elif isinstance(target, (np.ndarray, list, tuple)):
            target = np.array(target).tolist()
            where += ' AND b.bodyId=tid'
            if not pre_with:
                pre_with = f'WITH {target} AS targetIds'
                pre_unwind += 'UNWIND targetIds AS tid'
            else:
                pre_with += f', {target} AS targetIds'
                pre_unwind += '\nUNWIND targetIds AS tid'
        else:
            where += f' AND b.bodyId={target}'

    ret = ['a.bodyId AS source', 'b.bodyId AS target', 'count(*) AS synapses']

    if add_props:
        ret += [f'a.{p} AS source_{p}' for p in add_props]
        ret += [f'b.{p} AS target_{p}' for p in add_props]

    ret = ', '.join(ret)
        
    # MATCH (n:`hemibrain_Neuron`)-[:ConnectsTo]-(m:`hemibrain_Neuron`),
    #       (n)-[:Contains]->(nss:SynapseSet),
    #       (m)-[:Contains]->(mss:SynapseSet),
    #       (nss)-[:ConnectsTo]-(mss),
    #       (mss)-[:Contains]->(ms:Synapse),
    #       (nss)-[:Contains]->(ns:Synapse),
    #       (ns)-[:SynapsesTo]-(ms)
    # WHERE n.`SNP(R)` AND m.`SNP(R)` AND ns.`SNP(R)` AND n.bodyId=294424196
    # RETURN n.bodyId AS source, m.bodyId AS target, count(*) AS synapses

    cypher = f"""\
             {pre_with} {pre_unwind}
             MATCH (a:`{dataset}_{datatype}`)<-[:From]-(c:ConnectionSet)-[:To]->(b:`{dataset}_{datatype}`), (c)-[:Contains]->(s:Synapse)
             WHERE {where}
             RETURN {ret}
             """

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

    where = ['(s.type="post")']
    if not isinstance(roi, type(None)):
        if not isinstance(roi, str):
            raise TypeError(f'Expected ROI as str, got "{type(roi)}"')
        if roi.startswith('~'):
            where.append(f'NOT (exists(s.`{roi[1:]}`))')
        else:
            where.append(f'(exists(s.`{roi}`))')

    pre_with = []
    pre_unwind = []

    if not isinstance(source, type(None)):
        if isinstance(source, str):
            if source.isnumeric():
                where.append(f'a.bodyId={source}')
            else:
                where.append(f'a.name=~"{source}"')
        elif isinstance(source, (np.ndarray, list, tuple)):
            source = np.array(source).tolist()
            where.append('a.bodyId=sid')
            pre_with.append(f'{source} AS sourceIds')
            pre_unwind.append('sourceIds AS sid')
        else:
            where.append(f'a.bodyId={source}')

    if not isinstance(target, type(None)):
        if isinstance(target, str):
            if target.isnumeric():
                where.append(f'b.bodyId={target}')
            else:
                where.append(f'b.name=~"{target}"')
        elif isinstance(target, (np.ndarray, list, tuple)):
            target = np.array(target).tolist()
            pre_with.append(f'{target} AS targetIds')
            if not pre_with:
                # Only unwind targets if we aren't already unwinding sources
                pre_unwind.append('targetIds AS tid')
                where.append('b.bodyId=tid')
            else:
                where.append('b.bodyId IN targetIds')
        else:
            where.append(f'b.bodyId={target}')

    ret = ['a.bodyId AS source', 'b.bodyId AS target', 'count(*) AS synapses']

    if add_props:
        ret += [f'a.{p} AS source_{p}' for p in add_props]
        ret += [f'b.{p} AS target_{p}' for p in add_props]

    pre_with = ', '.join(pre_with),
    pre_unwind = ', '.join(pre_unwind),
    where = ' AND '.join(where),
    ret = ', '.join(ret)
    
    cypher = f"""
             WITH {pre_with}
             UNWIND {pre_unwind}
             MATCH (a:`{dataset}_{datatype}`)<-[:From]-(c:ConnectionSet)-[:To]->(b:`{dataset}_{datatype}`), (c)-[:Contains]->(s:Synapse)
             WHERE {where}
             RETURN {ret}
             """

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
            where = f'bodyId={x}'
        else:
            where = f'name=~"{x}"'
    elif isinstance(x, (np.ndarray, list, tuple)):
        where = 'bodyId=bid'
        pre = f'WITH {list(x)} AS bodyIds UNWIND bodyIds AS bid'
    else:
        where = f'bodyId={x}'

    cypher = f"""
             {pre}
             MATCH (n:`{dataset}_{datatype}`)-[:Contains]->(ss:SynapseSet),
                   (ss)-[:Contains]->(s:Synapse)
             WHERE n.{where}
             RETURN 'n.bodyId as bodyId, s'
             """

    # Get data
    r = fetch_custom(cypher, client=client, format='json')

    # Flatten Synapse data
    s = pd.io.json.json_normalize([s[1] for s in r['data']])
    s['bodyId'] = [s[0] for s in r['data']]

    return s.fillna(False)
