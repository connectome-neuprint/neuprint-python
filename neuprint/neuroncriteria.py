import re
import copy
import inspect
import functools
from textwrap import indent, dedent
from collections.abc import Iterable, Collection

import numpy as np
import pandas as pd

from .utils import make_args_iterable
from .client import inject_client

def neuroncriteria_args(*argnames):
    """
    Returns a decorator.
    For the given argument names, the decorator converts the
    arguments into NeuronCriteria objects via ``copy_as_neuroncriteria()``.
    
    If the decorated function also accepts a 'client' argument,
    that argument is used to initialize the NeuronCriteria.
    """
    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            callargs = inspect.getcallargs(f, *args, **kwargs)
            for name in argnames:
                callargs[name] = copy_as_neuroncriteria(callargs[name], callargs.get('client', None))
            return f(**callargs)

        wrapper.__signature__ = inspect.signature(f)
        return wrapper

    return decorator


def copy_as_neuroncriteria(obj, client=None):
    """
    If the given argument is a NeuronCriteria object, copy it.
    Otherwise, attempt to construct a NeuronCriteria object,
    using the argument as either the bodyId or the type AND instance.

    Rules:

        NC -> copy(NC)

        None -> NC()

        int -> NC(bodyId)
        [str,...] -> NC(bodyId)
        DataFrame['bodyId'] -> NC(bodyId)
        
        str -> NC(type, instance)
        [str, ...] -> NC(type, instance)

        [] -> Error
        [None] -> Error
        Anything else -> Error
        
    """
    if isinstance(obj, pd.DataFrame):
        assert 'bodyId' in obj.columns, \
            'If passing a DataFrame as NeuronCriteria, it must have "bodyId" column'
        return NeuronCriteria(bodyId=obj['bodyId'].values, client=client)

    if not isinstance(obj, Collection) or isinstance(obj, str):
        if obj is None:
            return NeuronCriteria(client=client)
        
        if isinstance(obj, NeuronCriteria):
            return copy.copy(obj)
    
        if isinstance(obj, str):
            return NeuronCriteria(type=obj, instance=obj, client=client)
        
        if np.issubdtype(type(obj), np.integer):
            return NeuronCriteria(bodyId=obj, client=client)

        raise RuntimeError(f"Can't auto-construct a NeuronCriteria from {obj}.  Please explicitly create one.")
    else:
        if len(obj) == 0:
            raise RuntimeError(f"Can't auto-construct a NeuronCriteria from {obj}.  Please explicitly create one.")
    
        if len(obj) == 1 and obj[0] is None:
            raise RuntimeError(f"Can't auto-construct a NeuronCriteria from {obj}.  Please explicitly create one.")
    
        if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.integer):
            return NeuronCriteria(bodyId=obj, client=client)
        
        item = [*filter(lambda item: item is not None, obj)][0]
        if np.issubdtype(type(item), np.integer):
            return NeuronCriteria(bodyId=obj, client=client)
    
        if isinstance(item, str):
            return NeuronCriteria(type=obj, instance=obj, client=client)
    
        raise RuntimeError(f"Can't auto-construct a NeuronCriteria from {obj}.  Please explicitly create one.")


class NeuronCriteria:
    """
    Specifies which fields to filter by when searching for a Neuron (or Segment).
    This class does not send queries itself, but you use it to specify search
    criteria for various query functions.
    
    Note:
        For simple queries involving only particular bodyId(s) or type(s)/instance(s),
        you can usually just pass the ``bodyId`` or ``type`` to the query function,
        without constructing a full ``NeuronCriteria``.
        
        .. code-block:: python
        
            from neuprint import fetch_neurons, NeuronCriteria as NC
        
            # Equivalent
            neuron_df, conn_df = fetch_neurons(NC(bodyId=329566174))
            neuron_df, conn_df = fetch_neurons(329566174)

            # Equivalent
            # (Criteria is satisfied if either type or instance matches.)
            neuron_df, conn_df = fetch_neurons(NC(type="OA-VPM3", instance="OA-VPM3"))
            neuron_df, conn_df = fetch_neurons("OA-VPM3")
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
                ``NeuronCriteria`` is used to construct cypher queries.
                To help catch errors (such as accidentally passing a ``type`` or
                ``instance`` name in the wrong argument position), we require that
                ``matchvar`` begin with a lowercase letter.

            bodyId (int or list of ints):
                List of bodyId values.

            instance (str or list of str):
                If ``regex=True``, then the instance will be matched as a regular expression.
                Otherwise, only exact matches are found. To search for neurons with no instance
                at all, use ``instance=[None]``. If both ``type`` and ``instance`` criteria are
                supplied, any neuron that matches EITHER criteria will match the overall criteria.

            type (str or list of str):
                If ``regex=True``, then the type will be matched as a regular expression.
                Otherwise, only exact matches are found. To search for neurons with no type
                at all, use ``type=[None]``. If both ``type`` and ``instance`` criteria are
                supplied, any neuron that matches EITHER criteria will match the overall criteria.

            regex (bool):
                If ``True``, the ``instance`` and ``type`` arguments will be interpreted as
                regular expressions, rather than exact match strings.

            status (str or list of str):
                Matches for the neuron ``status`` field.  To search for neurons with no status
                at all, use ``status=[None]``.

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
        # Validate that matchvar in various ways, to catch errors in which
        # the user has passed a bodyId or type, etc. in the wrong position.
        assert isinstance(matchvar, str), \
            (f"Bad matchvar argument (should be str): {matchvar}. "
             "Did you mean to pass this as bodyId, type, or instance name?")
        assert matchvar, "matchvar cannot be an empty string"
        assert re.match('^[a-z].*$', matchvar), \
            (f"matchvar must begin with a lowercase letter, not '{matchvar}'. "
             "Did you mean to pass this as a type or instance name?")
        assert re.match('^[a-zA-Z0-9]+$', matchvar), \
            (f"matchvar contains invalid characters: '{matchvar}'. "
             "Did you mean to pass this as a type or instance?")
        
        assert label in ('Neuron', 'Segment'), f"Invalid label: {label}"
        assert len(bodyId) == 0 or np.issubdtype(np.asarray(bodyId).dtype, np.integer), \
            "bodyId should be an integer or list of integers"

        assert not regex or len(instance) <= 1, \
            "Please provide only one regex pattern for instance"
        assert not regex or len(type) <= 1, \
            "Please provide only one regex pattern for type"

        if not regex and isinstance(instance, str) and len(instance) == 1:
            assert '.*' not in instance[0], \
                f"instance appears to be a regular expression ('{instance[0]}'), but you didn't pass regex=True"

        if not regex and isinstance(type, str) and len(type) == 1:
            assert '.*' not in type[0], \
                f"type appears to be a regular expression ('{type[0]}'), but you didn't pass regex=True"

        assert roi_req in ('any', 'all')

        assert min_roi_inputs <= 1 or inputRois, \
            "Can't stipulate min_roi_inputs without a list of inputRois"
        assert min_roi_outputs <= 1 or outputRois, \
            "Can't stipulate min_roi_outputs without a list of outputRois"

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


    def __eq__(self, value):
        """
        Implement comparison between criteria.
        Note: 'matchvar' is not considered during the comparison.
        """
        if not isinstance(value, NeuronCriteria):
            return NotImplemented

        # Return True if it's the exact same object
        if self is value:
            return True

        # Compare attributes one by one
        # But don't count 'matchvar' as a parameter'.
        params = [#'matchvar',
                 'bodyId', 'instance', 'type', 'status',
                 'cropped', 'min_pre', 'min_post', 'rois', 'inputRois',
                 'outputRois', 'min_roi_inputs', 'min_roi_outputs',
                 'regex', 'label', 'roi_req']

        for at in params:
            me = getattr(self, at)
            other = getattr(value, at)

            # If not the same type, return False
            if type(me) != type(other):
                return False

            # If iterable (e.g. ROIs or body IDs) we don't care about order
            if isinstance(me, Iterable):
                if not all([v in other for v in me]):
                    return False
            elif me != other:
                return False
        # If all comparisons have passed, return True
        return True


    def __repr__(self):
        # Show all non-default constructor args
        s = f'NeuronCriteria("{self.matchvar}"'
        
        if len(self.bodyId):
            s += f", bodyId={list(self.bodyId)}"
        
        if len(self.instance) == 1:
            s += f', instance="{self.instance[0]}"'
        elif len(self.instance) > 1:
            s += f", instance={list(self.instance)}"
            
        if len(self.type) == 1:
            s += f', type="{self.type[0]}"'
        elif len(self.instance) > 1:
            s += f", type={list(self.type)}"
        
        if self.regex:
            s += ", regex=True"

        if len(self.status) == 1:
            s += f', status="{self.status[0]}"'
        elif len(self.instance) > 1:
            s += f", status={list(self.status)}"
        
        if self.cropped is not None:
            s += f", cropped={self.cropped}"

        if self.min_pre != 0:
            s += f", min_pre={self.min_pre}"

        if self.min_post != 0:
            s += f", min_post={self.min_post}"

        if self.rois:
            s += f", rois={list(self.rois)}"

        if self.inputRois:
            s += f", inputRois={list(self.inputRois)}"

        if self.outputRois:
            s += f", outputRois={list(self.outputRois)}"
        
        if self.min_roi_inputs != 1:
            s += f", min_roi_inputs={self.min_roi_inputs}"
                    
        if self.min_roi_outputs != 1:
            s += f", min_roi_outputs={self.min_roi_outputs}"

        if self.label != 'Neuron':
            s += f', label="{self.label}"'

        if self.roi_req != 'all':
            s += f', roi_req="{self.roi_req}"'

        s += ')'
        
        return s
    

    def basic_exprs(self):
        """
        Return the list of expressions that correspond
        to the members in this NeuronCriteria object.
        They're intended be combined (via 'AND') in
        the WHERE clause of a cypher query.
        """
        exprs = [self.bodyId_expr(), self.typeinst_expr(), self.status_expr(),
                 self.cropped_expr(), self.rois_expr(), self.pre_expr(), self.post_expr()]
        exprs = [*filter(None, exprs)]
        return exprs


    def bodyId_expr(self):
        return where_expr('bodyId', self.bodyId, False, self.matchvar)

    def typeinst_expr(self):
        """
        Unlike all other fields, type and instance OR'd together.
        Either match satisfies the criteria.
        """
        t = self.type_expr()
        i = self.instance_expr()
        
        if t and i:
            return f"({t} OR {i})"
        if t:
            return t
        if i:
            return i
        return ""

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
        
        if not vars:
            vars = (self.matchvar,)
        
        basic_cond = self.basic_conditions(*vars, comments=comments)
        roi_cond = self.directed_rois_condition(*vars, comments=comments)
        
        if roi_cond:
            combined = basic_cond + "\n\n" + roi_cond
        else:
            combined = basic_cond

        return indent(combined, prefix)[len(prefix):]
        

    @classmethod
    def combined_conditions(cls, neuron_conditions, vars=None, prefix=0, comments=True):
        """
        Combine the conditions from multiple NeuronCriteria into a single string,
        putting the "cheap" conditions first and the "expensive" conditions last.
        (That is, basic conditions first and the directed ROI conditions last.)
        """
        if isinstance(prefix, int):
            prefix = ' '*prefix
            
        if not vars:
            vars = [sc.matchvar for sc in neuron_conditions]
        
        basic_conds = [sc.basic_conditions(*vars, comments=comments) for sc in neuron_conditions]
        basic_conds = [*filter(None, basic_conds)]
        if not basic_conds:
            return ""
        
        basic_conds = '\n\n'.join(basic_conds)
        combined = basic_conds
        
        roi_conds = [sc.directed_rois_condition(*vars, comments=comments) for sc in neuron_conditions]
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


#: Same as ``NeuronCriteria``.  This name is deprecated, but kept for backwards compatibility.
SegmentCriteria = NeuronCriteria


@make_args_iterable(['values'])
def where_expr(field, values, regex=False, matchvar='n'):
    """
    Return an expression to match a particular
    field against a list of values, to be used
    within the WHERE clause.
    """
    assert not regex or len(values) <= 1, \
        f"Can't use regex mode with more than one value: {values}"

    if len(values) == 0:
        return ""

    if len(values) == 1:
        if values[0] is None:
            return f"NOT exists({matchvar}.{field})"
    
        if regex:
            return f"{matchvar}.{field} =~ '{values[0]}'"
    
        if isinstance(values[0], str):
            return f"{matchvar}.{field} = '{values[0]}'"
    
        return f"{matchvar}.{field} = {values[0]}"

    # list of values

    if None not in values:
        return f"{matchvar}.{field} in {[*values]}"

    # ['some_val', None, 'some_other']
    values = [*filter(lambda v: v is not None, values)]
    return f"{matchvar}.{field} in {[*values]} OR NOT exists({matchvar}.{field})"


