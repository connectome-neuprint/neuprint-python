import pandas as pd
import numpy as np

from .utils import eval_client, make_iterable


def fetch_custom(cypher, client=None):
    """ Fetch custom cypher.

    Parameters
    ----------
    cypher :    str
                Cypher to fetch.
    client :    neuprint.Client, optional
                If ``None`` will try using global client.

    Returns
    -------
    pandas.DataFrame
    """

    client = eval_client(client)

    return client.fetch_custom(cypher)


def custom_search(x, props=['bodyId', 'name'], logic='AND', dataset='hemibrain',
                  datatype='Neuron', client=None):
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

    client = eval_client(client)

    x = make_iterable(x)

    where = logic.join(['n.{}'.format(s) for s in x])
    ret = ','.join(['n.{0} AS {0}'.format(p) for p in props])

    cypher = """
             MATCH (n :`{dataset}-{datatype}`)
             WHERE {where}
             RETURN {ret}
             """.format(dataset=dataset, datatype=datatype, where=where,
                ret=ret)

    return client.fetch_custom(cypher)


def fetch_neurons_in_roi(roi, dataset='hemibrain', datatype='Neuron',
                         logic='AND', add_props=None, client=None):
    """ Fetch all neurons within given ROI.

    Parameters
    ----------
    roi :       str | iterable
                ROI(s) (e.g. "LH") to query. See ``neuprint.Client.fetch_datasets``
                for available ROIs.
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

    client = eval_client(client)

    roi = make_iterable(roi)

    cypher = """
             MATCH (n :`{dataset}-{datatype}`)
             WHERE {where} WITH n AS n, apoc.convert.fromJsonMap(n.roiInfo) AS roiInfo
             RETURN n.bodyId AS bodyId, n.size AS size, n.status AS status,
                    n.pre AS pre, n.post AS post, {roiPre}, {roiPost}
             """.format(dataset=dataset, datatype=datatype,
                        roiPre=','.join(['roiInfo.{0}.pre as pre_{0}'.format(r) for r in roi]),
                        roiPost=','.join(['roiInfo.{0}.post as post_{0}'.format(r) for r in roi]),
                        where=logic.join(['(n.`{}`=true)'.format(r) for r in roi]))

    if add_props:
        cypher += ','
        cypher += ','.join(['n.{0} AS {0}'.format(p) for p in add_props])

    return client.fetch_custom(cypher)


def find_neurons(x, dataset='hemibrain', datatype='Neuron', add_props=None,
                 client=None):
    """ Find neurons by name or body ID.

    Parameters
    ----------
    x :         str
                Search string. Can be body ID, neuron name or wildcard names
                (e.g. "MBON.*"). Accepts regex.
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

    if isinstance(x, str):
        if x.isnumeric():
            where = 'bodyId={}'.format(x)
        else:
            where = 'name=~"{}"'.format(x)
    else:
        where = 'bodyId={}'.format(x)

    props = ['bodyId', 'name', 'size', 'status', 'pre', 'post']

    if add_props:
        props += list(add_props)
        props = list(set(props))

    return custom_search(where, props=props, dataset=dataset,
                         datatype=datatype, client=client)


def fetch_connectivity(x, dataset='hemibrain', datatype='Neuron', add_props=None,
                 client=None):
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

    if isinstance(x, str):
        if x.isnumeric():
            where = 'bodyId={}'.format(x)
        else:
            where = 'name=~"{}"'.format(x)
    elif isinstance(x, (np.ndarray, list, tuple)):
        where = 'bodyId IN {}'.format(x)
    else:
        where = 'bodyId={}'.format(x)

    ret = ['m.name AS name1', 'n.name AS name2', 'e.weight AS weight',
           'n.bodyId AS body2', 'id(m) AS id1', 'id(n) AS id2',
           'id(startNode(e)) AS pre_id', 'm.bodyId AS body1',
           'e.weightHP AS WeightHP']

    if add_props:
        ret += ['n.{} as {}'.format(p, p) for p in add_props]

    client = eval_client(client)

    cypher = """
             MATCH (m:`{dataset}-{datatype}`)-[e:ConnectsTo]-(n)
             WHERE m.{where}
             RETURN {ret}
             """.format(dataset=dataset, datatype=datatype, where=where,
                        ret=', '.join(ret))

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


def fetch_connectivity_in_roi(roi, source=None, target=None, dataset='hemibrain',
                              datatype='Neuron', add_props=None, client=None):
    """ Fetch connectivity within ROI between given neuron(s).

    Parameters
    ----------
    roi :       str
                ROI(s) to filter for.
    source :    str | int | iterable | None, optional
                Source neurons. Can be body ID, neuron name or wildcard names
                (e.g. "MBON.*"). Accepts regex. Body IDs can be given as
                list. If ``None`` will get all inputs to ``target``.
    target :    str | int | iterable | None
                Target neurons. If ``None`` will get all outputs of ``sources``.
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

    if isinstance(source, type(None)) and isinstance(target, type(None)):
        raise ValueError('source and target must not both be "None"')

    where ='(exists(s.`{}`)) AND (s.type="post")'.format(roi)

    if source:
        if isinstance(source, str):
            if source.isnumeric():
                where += ' AND a.bodyId={}'.format(source)
            else:
                where += ' AND a.name=~"{}"'.format(source)
        elif isinstance(source, (np.ndarray, list, tuple)):
            where += ' AND a.bodyId IN {}'.format(source)
        else:
            where += ' AND a.bodyId={}'.format(source)

    if target:
        if isinstance(target, str):
            if target.isnumeric():
                where += ' AND b.bodyId={}'.format(target)
            else:
                where += ' AND b.name=~"{}"'.format(target)
        elif isinstance(target, (np.ndarray, list, tuple)):
            where += ' AND b.bodyId IN {}'.format(target)
        else:
            where += ' AND b.bodyId={}'.format(target)

    ret = ['a.bodyId AS source', 'b.bodyId AS target', 'count(*) AS synapses']

    if add_props:
        ret += ['a.{} AS source_{}'.format(p, p) for p in add_props]
        ret += ['b.{} AS target_{}'.format(p, p) for p in add_props]

    client = eval_client(client)

    cypher = """
             MATCH (a:`{dataset}-{datatype}`)<-[:From]-(c:ConnectionSet)-[:To]->(b:`{dataset}-{datatype}`), (c)-[:Contains]->(s:Synapse)
             WHERE {where}
             RETURN {ret}
             """.format(dataset=dataset, datatype=datatype, where=where,
                        ret=', '.join(ret))

    return cypher

    # Fetch data
    data = client.fetch_custom(cypher)

    return data.sort_values('synapses', ascending=False)

