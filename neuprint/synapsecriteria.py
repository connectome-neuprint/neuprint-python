from textwrap import indent, dedent

from .utils import make_args_iterable
from .client import inject_client

class SynapseCriteria:
    """
    Specifies which fields to filter by when searching for a Synapses.
    This class does not send queries itself, but you use it to specify search
    criteria for various query functions.
    """

    @inject_client
    @make_args_iterable(['rois'])
    def __init__(self, matchvar='s', *, rois=None, type=None, confidence=0.0, primary_only=False, client=None):
        """
        Except for ``matchvar``, all parameters must be passed as keyword arguments.

        Args:
            matchvar (str):
                An arbitrary cypher variable name to use when this
                ``SynapseCriteria`` is used to construct cypher queries.
        
            rois (str or list):
                Optional.
                If provided, limit the results to synapses that reside within the given roi(s).
    
            type:
                If provided, limit results to either 'pre' or 'post' synapses.
    
            confidence (float, 0.0-1.0):
                Limit results to synapses of at least this confidence rating.
    
            primary_only (boolean):
                If True, only include primary ROI names in the results.

                Note:
                    This parameter does NOT filter by ROI. (See the ``rois`` argument for that.)
                    It merely determines whether or each synapse should be associated with exactly
                    one ROI in the query output, or with multiple ROIs (one for every non-primary
                    ROI the synapse intersects).
    
            client:
                Used to validate ROI names.
                If not provided, the global default :py:class:`.Client` will be used.
        """
        unknown_rois = {*rois} - {*client.all_rois}
        assert not unknown_rois, f"Unrecognized synapse rois: {unknown_rois}"

        assert type in ('pre', 'post', None), \
            f"Invalid synapse type: {type}.  Choices are 'pre' and 'post'."

        self.matchvar = matchvar
        self.rois = rois
        self.type = type
        self.confidence = confidence
        self.primary_only = primary_only
        
    
    def condition(self, *vars, prefix='', comments=True):
        """
        Construct a cypher WITH..WHERE clause to filter for synapse criteria.
        
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
        
        roi_expr = conf_expr = type_expr = ""
        if self.rois:
            roi_expr = '(' + ' OR '.join([f'{self.matchvar}.`{roi}`' for roi in self.rois]) + ')'

        if self.confidence:
            conf_expr = f'({self.matchvar}.confidence > {self.confidence})'

        if self.type:
            type_expr = f"({self.matchvar}.type = '{self.type}')"

        exprs = [*filter(None, [roi_expr, conf_expr, type_expr])]

        if not exprs:
            return ""

        cond = dedent(f"""\
            WITH {', '.join(vars)}
            WHERE {' AND '.join(exprs)}
            """)

        if comments:
            cond = f"// -- Filter synapse '{self.matchvar}' --\n" + cond

        cond = indent(cond, prefix)[len(prefix):]
        return cond


    def __eq__(self, other):
        return (    (self.matchvar == other.matchvar)
                and (self.rois == other.rois)
                and (self.type == other.type)
                and (self.confidence == other.confidence)
                and (self.primary_only == other.primary_only))


    def __repr__(self):
        s = f"SynapseCriteria('{self.matchvar}'"
        
        args = []
        
        if self.rois:
            args.append("rois=[" + ", ".join(f"'{roi}'" for roi in self.rois) + "]")
        
        if self.type:
            args.append(f"type='{self.type}'")
        
        if self.confidence:
            args.append(f"confidence={self.confidence}")
        
        if self.primary_only:
            args.append("primary_only=True")
        
        if args:
            s += ', ' + ', '.join(args)
        
        s += ")"
        return s
