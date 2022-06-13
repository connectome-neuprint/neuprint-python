from asciitree import LeftAligned

from ..client import inject_client
from .general import fetch_meta


@inject_client
def fetch_all_rois(*, client=None):
    """
    List all ROIs in the dataset.
    """
    meta = fetch_meta(client=client)
    return _all_rois_from_meta(meta)


def _all_rois_from_meta(meta):
    rois = {*meta['roiInfo'].keys()}

    if meta['dataset'] == 'hemibrain':
        # These ROIs are special:
        # For historical reasons, they exist as tags,
        # but are not (always) listed in the Meta roiInfo.
        rois |= {'FB-column3', 'AL-DC3'}
        rois |= {f"AL-DC{i}(R)" for i in [1,2,3,4]}

    return sorted(rois)


@inject_client
def fetch_primary_rois(*, client=None):
    """
    List 'primary' ROIs in the dataset.
    Primary ROIs do not overlap with each other.
    """
    q = "MATCH (m:Meta) RETURN m.primaryRois as rois"
    rois = client.fetch_custom(q)['rois'].iloc[0]
    return sorted(rois)


def fetch_roi_hierarchy(include_subprimary=True, mark_primary=True, format='dict', *, client=None):
    """
    Fetch the ROI hierarchy nesting relationships.

    Most ROIs in neuprint are part of a hierarchy of nested regions.
    The structure of the hierarchy is stored in the dataset metadata,
    and can be retrieved with this function.

    Args:
        include_subprimary:
            If True, all hierarchy levels are included in the output.
            Otherwise, the hierarchy will only go as deep as necessary to
            cover all "primary" ROIs, but not any sub-primary ROIs that
            are contained within them.

        mark_primary:
            If True, append an asterisk (``*``) to the names of
            "primary" ROIs in the hierarchy.
            Primary ROIs do not overlap with each other.

        format:
            Either ``"dict"``, ``"text"``, or ``nx``.
            Specifies whether to return the hierarchy as a `dict`, or as
            a printable text-based tree, or as a ``networkx.DiGraph``
            (requires ``networkx``).

    Returns:
        Either ``dict``, ``str``, or ``nx.DiGraph``,
        depending on your chosen ``format``.

    Example:

        .. code-block:: ipython

            In [1]: from neuprint.queries import fetch_roi_hierarchy
               ...:
               ...: # Print the first few nodes of the tree -- you get the idea
               ...: roi_tree_text = fetch_roi_hierarchy(False, True, 'text')
               ...: print(roi_tree_text[:180])
            hemibrain
             +-- AL(L)*
             +-- AL(R)*
             +-- AOT(R)
             +-- CX
             |   +-- AB(L)*
             |   +-- AB(R)*
             |   +-- EB*
             |   +-- FB*
             |   +-- NO*
             |   +-- PB*
             +-- GC
             +-- GF(R)
             +-- GNG*
             +-- INP
             |
    """
    assert format in ('dict', 'text', 'nx')
    meta = fetch_meta(client=client)
    hierarchy = meta['roiHierarchy']
    primary_rois = {*meta['primaryRois']}

    def insert(h, d):
        name = h['name']
        is_primary = (name in primary_rois)
        if mark_primary and is_primary:
            name += "*"

        d[name] = {}

        if 'children' not in h:
            return

        if is_primary and not include_subprimary:
            return

        for c in sorted(h['children'], key=lambda c: c['name']):
            insert(c, d[name])

    d = {}
    insert(hierarchy, d)

    if format == 'dict':
        return d

    if format == "text":
        return LeftAligned()(d)

    if format == 'nx':
        import networkx as nx
        g = nx.DiGraph()

        def add_nodes(parent, d):
            for k in d.keys():
                g.add_edge(parent, k)
                add_nodes(k, d[k])
        add_nodes('hemibrain', d['hemibrain'])
        return g
