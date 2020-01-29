from textwrap import indent, dedent
import numpy as np
from .utils import make_args_iterable, where_expr
from .client import inject_client

class SegmentCriteria:
    """
    Specifies which fields to filter by when searching for a Segment (or Neuron).
    This class does not send queries itself, but you use it to specify search
    criteria for various query functions.
    """
    
    @inject_client
    @make_args_iterable(['bodyId', 'instance', 'type', 'status', 'rois', 'inputRois', 'outputRois'])
    def __init__( self, matchvar='n', *,
                  bodyId=None, instance=None, type=None, regex=False,
                  status=None, cropped=None,
                  min_pre=0, min_post=0,
                  rois=None, inputRois=None, outputRois=None, min_roi_inputs=1, min_roi_outputs=1,
                  label='Neuron', roi_req='all',
                  client=None ):
        """
        Except for ``matchvar``, all parameters must be passed as keyword arguments.
        
        .. note::
        
            **Options for specifying ROI criteria**
        
            The ``rois`` argument merely matches neurons that intersect the given ROIs at all
            (without distinguishing between inputs and outputs).
            
            The ``inputRois`` and ``outputRois`` arguments allow you to put requirements
            on whether or not neurons have inputs or outputs in the listed ROIs.
            It results a more expensive query, but its more powerful.
            It also enables you to require a minimum number of connections in the given
            ``inputRois`` or ``outputRois`` using the ``min_roi_inputs`` and ``min_roi_outputs``
            criteria.
            
            In either case, use use ``roi_req`` to specify whether a neuron must match just
            one (``any``) of the listed ROIs, or ``all`` of them.
        
        Args:
            matchvar (str):
                An arbitrary cypher variable name to use when this
                ``SegmentCriteria`` is used to construct cypher queries.

            bodyId (int or list of ints):
                List of bodyId values.

            instance (str or list of str):
                If ``regex=True``, then the instance will be matched as a regular expression.
                Otherwise, only exact matches are found.

            type (str or list of str):
                If ``regex=True``, then the type will be matched as a regular expression.
                Otherwise, only exact matches are found.

            regex (bool):
                If ``True``, the ``instance`` and ``type`` arguments will be interpreted as
                regular expressions, rather than exact match strings.

            status (str or list of str):
            
            cropped (bool):
                If given, restrict results to neurons that are cropped or not.

            min_pre (int):
                Exclude neurons that don't have at least this many t-bars (outputs) overall,
                regardless of how many t-bars exist in any particular ROI. 

            min_post (int):
                Exclude neurons that don't have at least this many PSDs (inputs) overall,
                regardless of how many PSDs exist in any particular ROI.
            
            rois (str or list of str):
                ROIs that merely intersect the neuron, without specifying whether
                they're intersected by input or output synapses.
                If not provided, will be auto-set from ``inputRois`` and ``outputRois``.
            
            inputRois (str or list of str):
                Only Neurons which have inputs in EVERY one of the given ROIs will be matched.
                ``regex`` does not apply to this parameter.

            outputRois (str or list of str):
                Only Neurons which have outputs in EVERY one of the given ROIs will be matched.
                ``regex`` does not apply to this parameter.

            min_roi_inputs (int):
                How many input (post) synapses a neuron must have in each ROI to satisfy the
                ``inputRois`` criteria.  Can only be used if you provided ``inputRois``.

            min_roi_outputs (int):
                How many output (pre) synapses a neuron must have in each ROI to satisfy the
                ``outputRois`` criteria.   Can only be used if you provided ``outputRois``.

            roi_req (Either ``'any'`` or ``'all'``):
                Whether a neuron must intersect all of the listed input/output ROIs, or any of the listed input/output ROIs.
                When using 'any', each neuron must still match at least one input AND at least one output ROI.

            label (Either ``'Neuron'`` or ``'Segment'``):
                Which node label to match with.
                (In neuprint, all ``Neuron`` nodes are also ``Segment`` nodes.)

            client (:py:class:`neuprint.client.Client`):
                Used to validate ROI names.
                If not provided, the global default ``Client`` will be used.
        """
        assert label in ('Neuron', 'Segment'), f"Invalid label: {label}"
        assert len(bodyId) == 0 or np.issubdtype(np.asarray(bodyId).dtype, np.integer), \
            "bodyId should be an integer or list of integers"
        
        assert not regex or len(instance) <= 1, "Please provide only one regex pattern for instance"
        assert not regex or len(type) <= 1, "Please provide only one regex pattern for type"
        assert roi_req in ('any', 'all')

        assert min_roi_inputs <= 1 or inputRois, \
            "Can't stipulate min_roi_inputs without a list of inputRois"
        assert min_roi_outputs <= 1 or outputRois, \
            "Can't stipulate min_roi_inputs without a list of outputRois"

        # If the user provided both intersecting rois and input/output rois,
        # force them to make the intersecting set a superset of the others.
        rois = {*rois}
        inputRois = {*inputRois}
        outputRois = {*outputRois}
        assert not rois or rois >= {*inputRois}, "Queried intersecting rois must be a superset of the inputRois"
        assert not rois or rois >= {*outputRois}, "Queried intersecting rois must be a superset of the outputRois"

        # Make sure intersecting is a superset of inputRois and outputRois
        rois |= {*inputRois, *outputRois}
        
        # Verify ROI names against known ROIs.
        neuprint_rois = {*client.all_rois}
        unknown_input_rois = inputRois - neuprint_rois
        if unknown_input_rois:
            raise RuntimeError(f"Unrecognized input ROIs: {unknown_input_rois}")
    
        unknown_output_rois = outputRois - neuprint_rois
        if unknown_output_rois:
            raise RuntimeError(f"Unrecognized output ROIs: {unknown_output_rois}")

        unknown_generic_rois = rois - neuprint_rois
        if unknown_generic_rois:
            raise RuntimeError(f"Unrecognized output ROIs: {unknown_generic_rois}")

        self.matchvar = matchvar
        self.bodyId = bodyId
        self.instance = instance
        self.type = type
        self.status = status
        self.cropped = cropped
        self.min_pre = min_pre
        self.min_post = min_post
        self.rois = rois
        self.inputRois = inputRois
        self.outputRois = outputRois
        self.min_roi_inputs = min_roi_inputs
        self.min_roi_outputs = min_roi_outputs
        self.regex = regex
        self.label = label
        self.roi_req = roi_req
    
    def basic_exprs(self):
        """
        Return the list of expressions that correspond
        to the members in this SegmentWhere object.
        They're intended be combined (via 'AND') in
        the WHERE clause of a cypher query. 
        """
        exprs = [self.bodyId_expr(), self.instance_expr(), self.type_expr(), self.status_expr(),
                 self.cropped_expr(), self.rois_expr(), self.pre_expr(), self.post_expr()]
        exprs = [*filter(None, exprs)]
        return exprs


    def bodyId_expr(self):
        return where_expr('bodyId', self.bodyId, False, self.matchvar)

    def instance_expr(self):
        return where_expr('instance', self.instance, self.regex, self.matchvar)

    def type_expr(self):
        return where_expr('type', self.type, self.regex, self.matchvar)

    def status_expr(self):
        return where_expr('status', self.status, False, self.matchvar)

    def cropped_expr(self):
        if self.cropped is None:
            return ""

        if self.cropped:
            return f"{self.matchvar}.cropped"
        else:
            # Not all neurons have the 'cropped' tag,
            # so simply checking for False values isn't enough.
            # Must check exists().
            return f"(NOT {self.matchvar}.cropped OR NOT exists({self.matchvar}.cropped))"

    def rois_expr(self):
        if len(self.rois) == 0:
            return ""
        
        rois = sorted(self.rois)
        roi_logic = {'any': 'OR', 'all': 'AND'}[self.roi_req]
        return "(" + f" {roi_logic} ".join(f"{self.matchvar}.`{roi}`" for roi in rois) + ")"


    def pre_expr(self):
        if self.min_pre:
            return f"{self.matchvar}.pre >= {self.min_pre}"
        else:
            return ""

    def post_expr(self):
        if self.min_post:
            return f"{self.matchvar}.post >= {self.min_post}"
        else:
            return ""


    def all_conditions(self, *vars, prefix=0, comments=True):
        if isinstance(prefix, int):
            prefix = ' '*prefix
        
        basic_cond = self.basic_conditions(*vars, comments=comments)
        roi_cond = self.directed_rois_condition(*vars, comments=comments)
        
        if roi_cond:
            combined = basic_cond + "\n\n" + roi_cond
        else:
            combined = basic_cond

        return indent(combined, prefix)[len(prefix):]
        

    @classmethod
    def combined_conditions(cls, segment_conditions, vars=None, prefix=0, comments=True):
        """
        Combine the conditions from multiple SegmentCriteria into a single string,
        putting the "cheap" conditions first and the "expensive" conditions last.
        (That is, basic conditions first and the directed ROI conditions last.)
        """
        if isinstance(prefix, int):
            prefix = ' '*prefix
            
        if not vars:
            vars = [sc.matchvar for sc in segment_conditions]
        
        
        basic_conds = [sc.basic_conditions(*vars, comments=comments) for sc in segment_conditions]
        basic_conds = [*filter(None, basic_conds)]
        if not basic_conds:
            return ""
        
        basic_conds = '\n\n'.join(basic_conds)
        combined = basic_conds
        
        roi_conds = [sc.directed_rois_condition(*vars, comments=comments) for sc in segment_conditions]
        roi_conds = [*filter(None, roi_conds)]
        if roi_conds:
            roi_conds = '\n\n'.join(roi_conds)
            combined = basic_conds + "\n\n" + roi_conds
        
        return indent(combined, prefix)[len(prefix):]
        

    def basic_conditions(self, *vars, prefix=0, comments=True):
        """
        Construct a WHERE clause based on the basic conditions
        in this criteria (i.e. everything except for the "directed ROI" conditions.)
        """
        exprs = self.basic_exprs()
        if not exprs:
            return ""

        if isinstance(prefix, int):
            prefix = prefix*' '
            
        # Build WHERE clause by combining exprs for each field
        clauses = ""
        if comments:
            clauses += f"// -- Basic conditions for segment '{self.matchvar}' --\n"
            
        if vars:
            clauses += f"WITH {', '.join(vars)}\n"
        
        clauses += "WHERE\n"
        clauses += f"  "
        clauses += f"\n  AND ".join(exprs)

        return indent(clauses, prefix)[len(prefix):]


    def directed_rois_condition(self, *vars, prefix=0, comments=True):
        """
        Construct the ```WITH...WHERE``` statements that apply the "directed ROI"
        conditions specified by this criteria's ``inputRois`` and ``outputRois``
        members.
        
        These conditions are expensive to evaluate, so it's usually a good
        idea to position them LAST in your cypher query, once the result set
        has already been narrowed down by eariler filters.
        """
        if not self.inputRois and not self.outputRois:
            return ""

        if isinstance(prefix, int):
            prefix = prefix*' '

        if len(self.inputRois) == 0:
            min_input_matches = 0
        elif self.roi_req == 'any':
            min_input_matches = 1
        elif self.roi_req == 'all':
            min_input_matches = 'size(inputRois)'
        else:
            assert False

        if len(self.outputRois) == 0:
            min_output_matches = 0
        elif self.roi_req == 'any':
            min_output_matches = 1
        elif self.roi_req == 'all':
            min_output_matches = 'size(outputRois)'
        else:
            assert False

        if vars:
            assert self.matchvar in vars, "Pass all match vars, including the one that belongs to this criteria"
            vars = ', '.join(vars)
        else:
            vars = self.matchvar
         
        conditions = dedent(f"""\
            // -- Directed ROI conditions for segment '{self.matchvar}' --
            WITH {vars},
                 {[*self.inputRois]} as inputRois,
                 {[*self.outputRois]} as outputRois,
                 apoc.convert.fromJsonMap({self.matchvar}.roiInfo) as roiInfo
    
            // Check input ROIs (segment '{self.matchvar}')
            UNWIND keys(roiInfo) as roi
            WITH {vars}, roi, roiInfo, inputRois, outputRois, roiInfo[roi]['post'] as roi_post
            ORDER BY roi
            // No filter if no input ROIs were specified, otherwise select the ones that meet the reqs
            WHERE {min_input_matches} = 0 OR (roi in inputRois AND roi_post >= {self.min_roi_inputs})
            WITH {vars}, roiInfo, inputRois, outputRois, collect(roi) as matchingInputRois, size(collect(roi)) as numMatchingInputRois
            WHERE numMatchingInputRois >= {min_input_matches}
    
            // Check output ROIs (segment '{self.matchvar}')
            UNWIND keys(roiInfo) as roi
            WITH {vars}, roi, roiInfo, inputRois, outputRois, matchingInputRois, roiInfo[roi]['pre'] as roi_pre
            ORDER BY roi
            // No filter if no output ROIs were specified, otherwise select the ones that meet the reqs
            WHERE {min_output_matches} = 0 OR (roi in outputRois AND roi_pre >= {self.min_roi_outputs})
            WITH {vars}, inputRois, outputRois, matchingInputRois, collect(roi) as matchingOutputRois, size(collect(roi)) as numMatchingOutputRois
            WHERE numMatchingOutputRois >= {min_output_matches}
            """)
            #RETURN n, matchingInputRois, matchingOutputRois

        if not comments:
            conditions = '\n'.join(filter(lambda s: '//' not in s, conditions.split('\n')))

        return indent(conditions, prefix)[len(prefix):]


