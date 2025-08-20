from textwrap import indent, dedent

from ..utils import ensure_list_args, cypher_identifier
from ..client import inject_client


class MitoCriteria:
    """
    Mitochondrion selection criteria.

    Specifies which fields to filter by when searching for mitochondria.
    This class does not send queries itself, but you use it to specify search
    criteria for various query functions.
    """

    @inject_client
    @ensure_list_args(['rois'])
    def __init__(self, matchvar='m', *, rois=None, mitoType=None, size=0, primary_only=True, client=None):
        """
        Except for ``matchvar``, all parameters must be passed as keyword arguments.

        Args:
            matchvar (str):
                An arbitrary cypher variable name to use when this
                ``MitoCriteria`` is used to construct cypher queries.

            rois (str or list):
                Optional.
                If provided, limit the results to mitochondria that reside within the given roi(s).

            mitoType:
                If provided, limit the results to mitochondria of the specified type.
                Either ``dark``, ``medium``, or ``light``.
                (Neuroglancer users: Note that in the hemibrain mito segmentation, ``medium=3``)

            size:
                Specifies a minimum size (in voxels) for mitochondria returned in the results.

            primary_only (boolean):
                If True, only include primary ROI names in the results.
                Disable this with caution.

                Note:
                    This parameter does NOT filter by ROI. (See the ``rois`` argument for that.)
                    It merely determines whether or not each mitochondrion should be associated with exactly
                    one ROI in the query output, or with multiple ROIs (one for every non-primary
                    ROI the mitochondrion intersects).

                    If you set ``primary_only=False``, then the table will contain duplicate entries
                    for each mito -- one per intersecting ROI.

            client:
                Used to validate ROI names.
                If not provided, the global default :py:class:`.Client` will be used.
        """
        unknown_rois = {*rois} - {*client.all_rois}
        assert not unknown_rois, f"Unrecognized mito rois: {unknown_rois}"

        mitoType = mitoType or None
        assert mitoType in (None, 'dark', 'medium', 'light'), \
            f"Invalid mitoType: {mitoType}."

        self.matchvar = matchvar
        self.rois = rois
        self.mitoType = mitoType
        self.size = size
        self.primary_only = primary_only

    def condition(self, *vars, prefix='', comments=True):
        """
        Construct a cypher WITH..WHERE clause to filter for mito criteria.

        Any match variables you wish to "carry through" for subsequent clauses
        in your query must be named in the ``vars`` arguments.
        """
        if not vars:
            vars = [self.matchvar]

        assert self.matchvar in vars, \
            ("Please pass all match vars, including the one that "
             f"belongs to this criteria ('{self.matchvar}').")

        if isinstance(prefix, int):
            prefix = ' '*prefix

        type_expr = f'{self.matchvar}.type = "mitochondrion"'
        roi_expr = size_expr = mitoType_expr = ""

        if self.rois:
            roi_expr = '(' + ' OR '.join([f'{self.matchvar}.{cypher_identifier(roi)}' for roi in self.rois]) + ')'

        if self.size:
            size_expr = f'({self.matchvar}.size >= {self.size})'

        if self.mitoType:
            mitoType_expr = f"({self.matchvar}.mitoType = {repr(self.mitoType)})"

        exprs = [*filter(None, [type_expr, roi_expr, size_expr, mitoType_expr])]

        if not exprs:
            return ""

        cond = dedent(f"""\
            WITH {', '.join(vars)}
            WHERE {' AND '.join(exprs)}
            """)

        if comments:
            cond = f"// -- Filter mito '{self.matchvar}' --\n" + cond

        cond = indent(cond, prefix)[len(prefix):]
        return cond


    def __eq__(self, other):
        return (    (self.matchvar == other.matchvar)
                and (self.rois == other.rois)
                and (self.mitoType == other.mitoType)
                and (self.size == other.size)
                and (self.primary_only == other.primary_only))


    def __repr__(self):
        s = f"MitoCriteria('{self.matchvar}'"

        args = []

        if self.rois:
            args.append("rois=[" + ", ".join(f"{repr(roi)}" for roi in self.rois) + "]")

        if self.mitoType:
            args.append(f"mitoType={repr(self.mitoType)}")

        if self.size:
            args.append(f"size={self.size}")

        if self.primary_only:
            args.append("primary_only=True")

        if args:
            s += ', ' + ', '.join(args)

        s += ")"
        return s
