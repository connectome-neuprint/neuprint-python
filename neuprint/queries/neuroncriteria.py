import re
import copy
import inspect
import functools
import collections.abc
from itertools import chain
from textwrap import indent, dedent
from collections.abc import Iterable, Collection

import numpy as np
import pandas as pd

from ..utils import make_args_iterable, IsNull, NotNull
from ..client import inject_client

NoneType = type(None)


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
        [int,...] -> NC(bodyId)
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

    if obj is None:
        return NeuronCriteria(client=client)

    if isinstance(obj, NeuronCriteria):
        return copy.copy(obj)

    if not isinstance(obj, Collection) or isinstance(obj, str):
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
    Neuron selection criteria.

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
    @make_args_iterable(['bodyId', 'instance', 'type', 'cellBodyFiber',
                         'status', 'statusLabel', 'rois', 'inputRois', 'outputRois',
                         'hemilineage', 'class_', 'exitNerve'])
    def __init__( self, matchvar='n', *,
                  bodyId=None, type=None, instance=None, regex='guess',
                  class_=None, somaSide=None, exitNerve=None, hemilineage=None,
                  cellBodyFiber=None,
                  status=None, statusLabel=None, cropped=None,
                  min_pre=0, min_post=0,
                  rois=None, inputRois=None, outputRois=None, min_roi_inputs=1, min_roi_outputs=1,
                  label=None, roi_req='all',
                  soma=None,
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

        .. note::

            **Matching against missing values (NULL)**

            To search for neurons which are missing given property entirely,
            you can use a list containing ``None``, or the special value ``neuprint.IsNull``.
            For example, to find neurons with no `type`, use ``type=[None]`` or ``type=IsNull``.

            **Matching against any value (NOT NULL)**

            To search for any non-null value, you can use ``neuprint.NotNull``. For
            example, to find neurons that have a type (no matter what the
            type is), use ``type=NotNull``.

        Args:
            matchvar (str):
                An arbitrary cypher variable name to use when this
                ``NeuronCriteria`` is used to construct cypher queries.
                To help catch errors (such as accidentally passing a ``type`` or
                ``instance`` name in the wrong argument position), we require that
                ``matchvar`` begin with a lowercase letter.

            bodyId (int or list of ints):
                List of bodyId values.

            type (str or list of str):
                Cell type.  Matches depend on the the ``regex`` argument.
                To search for neurons with no type at all, use ``type=[None]``.
                If both ``type`` and ``instance`` criteria are supplied, any neuron that
                matches EITHER criteria will match the overall criteria.

            instance (str or list of str):
                Cell instance (specific cell name).  Matches depend on the the ``regex`` argument.
                To search for neurons with no instance at all, use ``instance=[None]``.
                If both ``type`` and ``instance`` criteria are supplied, any neuron that
                matches EITHER criteria will match the overall criteria.

            regex (bool):
                If ``True``, the ``instance`` and ``type`` arguments will be interpreted as
                regular expressions, rather than exact match strings.
                If ``False``, only exact matches will be found.
                By default, the matching method will be automatically chosen by inspecting the
                ``type`` and ``instance`` strings.  If they contain regex-like characters,
                then we assume you intend regex matching. (You can see which method was chosen by
                checking the ``regex`` field after the ``NeuronCriteria`` is constructed.)

            class_ (str or list of str):
                Matches for the neuron ``class`` field.  To search for neurons
                with no class at all, use ``class_=[None]``.

            somaSide ('RHS' or 'LHS' or None):
                Matches for the neuron ``somaSide`` field.

            exitNerve (str or list of str):
                Matches for the neuron ``exitNerve`` field.  To search for neurons
                with no exit nerve at all, use ``exitNerve=[None]``.

            hemilineage (str or list of str):
                Matches for the neuron ``hemilineage`` field.  To search for neurons
                with no hemilineage at all, use ``hemilineage=[None]``.

            cellBodyFiber (str or list of str):
                Matches for the neuron ``cellBodyFiber`` field.  To search for neurons
                with no CBF at all, use ``cellBodyFiber=[None]``.

            status (str or list of str):
                Matches for the neuron ``status`` field.  To search for neurons with no status
                at all, use ``status=[None]``.

            statusLabel (str or list of str):
                Matches for the neuron ``statusLabel`` field. ``statusLabel`` is
                typically more finegrained than ``status``. To search for neurons
                with no status at all, use ``statusLabel=[None]``.

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

            label (Either ``'Neuron'`` or ``'Segment'``):
                Which node label to match with.
                (In neuprint, all ``Neuron`` nodes are also ``Segment`` nodes.)
                By default, ``'Neuron'`` is used, unless you provided a non-empty ``bodyId`` list.
                In that case, ``'Segment'`` is the default. (It's assumed you're really interested
                in the bodies you explicitly listed, whether or not they have the ``'Neuron'`` label.)

            roi_req (Either ``'any'`` or ``'all'``):
                Whether a neuron must intersect all of the listed input/output ROIs, or any of the listed input/output ROIs.
                When using 'any', each neuron must still match at least one input AND at least one output ROI.

            soma (Either ``True``, ``False``, or ``None``)
                If ``True``, only return neurons with a ``somaLocation``.
                If ``False``, return neurons without a ``somaLocation``.

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

        assert len(bodyId) == 0 or np.issubdtype(np.asarray(bodyId).dtype, np.integer), \
            "bodyId should be an integer or list of integers"

        if not label:
            if len(bodyId) == 0:
                label = 'Neuron'
            else:
                label = 'Segment'
        assert label in ('Neuron', 'Segment'), f"Invalid label: {label}"

        assert regex in (True, False, 'guess')

        for i in instance:
            assert isinstance(i, (str, NoneType)) or i in (IsNull, NotNull), \
                f'instance should be a string, IsNull, NotNull or None, got {i}'

        for t in type:
            assert isinstance(t, (str, NoneType)) or t in (IsNull, NotNull), \
                f'type should be a string, IsNull, NotNull or None, got {t}'

        if regex == 'guess':
            rgx = re.compile(r'[\\\.\?\[\]\+\^\$\*]')
            instance_is_regex = False
            for i in instance:
                instance_is_regex |= isinstance(i, str) and bool(rgx.search(i or ''))

            type_is_regex = False
            for t in type:
                type_is_regex |= isinstance(t, str) and bool(rgx.search(t or ''))

            regex = type_is_regex or instance_is_regex

        assert roi_req in ('any', 'all')

        assert min_roi_inputs <= 1 or inputRois, \
            "Can't stipulate min_roi_inputs without a list of inputRois"
        assert min_roi_outputs <= 1 or outputRois, \
            "Can't stipulate min_roi_outputs without a list of outputRois"

        assert soma in (True, False, None), \
            f"soma must be True, False or None, not {soma}"

        assert somaSide in ("RHS", "LHS", None), \
            f"somaSide must be 'LHS', 'RHS' or None, not {somaSide}"

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
        self.cellBodyFiber = cellBodyFiber
        self.status = status
        self.statusLabel = statusLabel
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
        self.soma = soma
        self.class_ = class_
        self.somaSide = somaSide
        self.exitNerve = exitNerve
        self.hemilineage = hemilineage

        self.list_props = ['bodyId', 'status', 'statusLabel', 'cellBodyFiber',
                           'hemilineage', 'exitNerve', 'class_']
        self.list_props_regex = ['type', 'instance']

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
                 'bodyId', 'instance', 'type', 'status', 'statusLabel',
                 'cropped', 'min_pre', 'min_post', 'rois', 'inputRois',
                 'outputRois', 'min_roi_inputs', 'min_roi_outputs',
                 'regex', 'label', 'roi_req', 'soma']

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

        if len(self.type) or len(self.instance):
            s += f", regex={self.regex}"

        if len(self.cellBodyFiber) == 1:
            s += f', cellBodyFiber="{self.cellBodyFiber[0]}"'
        elif len(self.instance) > 1:
            s += f", cellBodyFiber={list(self.cellBodyFiber)}"

        if len(self.status) == 1:
            s += f', status="{self.status[0]}"'
        elif len(self.instance) > 1:
            s += f", status={list(self.status)}"

        if len(self.statusLabel) == 1:
            s += f', statusLabel="{self.statusLabel[0]}"'
        elif len(self.instance) > 1:
            s += f", statusLabel={list(self.statusLabel)}"

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

        if self.soma is not None:
            s += f', soma="{self.soma}"'

        s += ')'

        return s

    MAX_LITERAL_LENGTH = 3
    assert MAX_LITERAL_LENGTH >= 3, \
        ("The logic in where_expr() assumes valuevars "
         "have length 3 (assuming one could be None).")

    def global_vars(self):
        exprs = {}

        for key in self.list_props:
            values = getattr(self, key)
            if len(values) > self.MAX_LITERAL_LENGTH:
                if key.endswith('_'):
                    key = key[:-1]
                values = [*filter(lambda s: s is not None, values)]
                var = f"{self.matchvar}_search_{key}"
                exprs[var] = (f"{[*values]} as {var}")

        for key in self.list_props:
            values = getattr(self, key)
            if not self.regex and len(values) > self.MAX_LITERAL_LENGTH:
                if key.endswith('_'):
                    key = key[:-1]
                values = [*filter(lambda s: s is not None, values)]
                var = f"{self.matchvar}_search_{key}"
                exprs[var] = (f"{[*values]} as {var}")

        return exprs

    def global_with(self, *vars, prefix=0):
        if isinstance(prefix, int):
            prefix = ' '*prefix

        if vars:
            carry_forward = [', '.join(vars)]
        else:
            carry_forward = []

        full_list = ',\n     '.join([*carry_forward, *self.global_vars().values()])
        if full_list:
            return indent('WITH ' + full_list, prefix)[len(prefix):]
        return ""

    def basic_exprs(self):
        """
        Return the list of expressions that correspond
        to the members in this NeuronCriteria object.
        They're intended be combined (via 'AND') in
        the WHERE clause of a cypher query.
        """
        exprs = [self.bodyId_expr(), self.typeinst_expr(), self.cbf_expr(),
                 self.status_expr(), self.statusLabel_expr(),
                 self.cropped_expr(), self.rois_expr(), self.pre_expr(), self.post_expr(),
                 self.soma_expr(), self.hemilineage_expr(), self.class_expr(),
                 self.exitNerve_expr(), self.somaSide_expr()]
        exprs = [*filter(None, exprs)]
        return exprs

    def _value_list_expr(self, key, value, regex=False):
        """
        Match key against a list of values. E.g. "bodyId in [1234, 5678]".
        """
        valuevar = None
        if not regex and len(value) > self.MAX_LITERAL_LENGTH:
            valuevar = f"{self.matchvar}_search_{key}"
        return where_expr(key, value, regex, self.matchvar, valuevar)

    def _single_value_expr(self, key, value):
        """
        Match against key/value:
            - True: key must exist
            - False: key must not exist
            - str: key must have given value
        """
        if value is None:
            return ""
        if not isinstance(value, bool):
            return f"{self.matchvar}.{key} = '{value}'"
        elif value:
            return f"{self.matchvar}.{key} IS NOT NULL"
        else:
            return f"{self.matchvar}.{key} IS NULL"

    def _tag_expr(self, key, value):
        """
        Match against tag, e.g. `.cropped`.
        Non-existing tags are counted as False.
        """
        if value is None:
            return ""

        if value:
            return f"{self.matchvar}.{key}"
        else:
            # Not all neurons might actually have the flag (e.g. `.cropped`),
            # so simply checking for False values isn't enough.
            # Must check exists().
            return f"(NOT {self.matchvar}.{key} OR NOT exists({self.matchvar}.{key}))"

    def _logic_tag_expr(self, tags, logic):
        """
        Match against logic list of tags, e.g. `.LH(R)` AND `.AL(R)`.
        """
        assert logic in ('AND', 'OR'), '`logic` must be either AND or OR'
        if len(tags) == 0:
            return ""

        tags = sorted(tags)
        return "(" + f" {logic} ".join(f"{self.matchvar}.`{v}`" for v in tags) + ")"

    def _gt_eq_expr(self, key, value):
        """
        Match against key/value being greater or equal.
        """
        if value:
            return f"{self.matchvar}.{key} >= {value}"
        else:
            return ""

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

    def bodyId_expr(self):
        return self._value_list_expr('bodyId', self.bodyId, False)

    def instance_expr(self):
        return self._value_list_expr('instance', self.instance, self.regex)

    def type_expr(self):
        return self._value_list_expr('type', self.type, self.regex)

    def cbf_expr(self):
        return self._value_list_expr('cellBodyFiber', self.cellBodyFiber, False)

    def status_expr(self):
        return self._value_list_expr('status', self.status, False)

    def statusLabel_expr(self):
        return self._value_list_expr('statusLabel', self.statusLabel, False)

    def hemilineage_expr(self):
        return self._value_list_expr('hemilineage', self.hemilineage, False)

    def exitNerve_expr(self):
        return self._value_list_expr('exitNerve', self.exitNerve, False)

    def class_expr(self):
        return self._value_list_expr('class', self.class_, False)

    def cropped_expr(self):
        return self._tag_expr('cropped', self.cropped)

    def rois_expr(self):
        return self._logic_tag_expr(self.rois,
                               {'any': 'OR', 'all': 'AND'}[self.roi_req])

    def pre_expr(self):
        return self._gt_eq_expr('pre', self.min_pre)

    def post_expr(self):
        return self._gt_eq_expr('post', self.min_post)

    def soma_expr(self):
        return self._single_value_expr('somaLocation', self.soma)

    def somaSide_expr(self):
        return self._single_value_expr('somaSide', self.somaSide)

    def all_conditions(self, *vars, prefix=0, comments=True):
        if isinstance(prefix, int):
            prefix = ' '*prefix

        vars = {*vars} | {self.matchvar, *self.global_vars().keys()}
        vars = (*vars,)

        basic_cond = self.basic_conditions(0, comments)
        if basic_cond:
            basic_cond = f"WHERE \n{basic_cond}"
            basic_cond = indent(basic_cond, '  ')[2:]

        roi_cond = self.directed_rois_condition(*vars, comments=comments)

        if roi_cond:
            combined = basic_cond + "\n\n" + roi_cond
        else:
            combined = basic_cond

        return indent(combined, prefix)[len(prefix):]

    @classmethod
    def combined_global_with(cls, neuron_conditions, vars=[], prefix=0):
        if isinstance(prefix, int):
            prefix = ' '*prefix

        if vars:
            carry_forward = [', '.join(vars)]
        else:
            carry_forward = []

        all_globals = chain(*(nc.global_vars().values() for nc in neuron_conditions))
        full_list = ',\n     '.join([*carry_forward, *all_globals])

        if full_list:
            return indent('WITH ' + full_list, prefix)[len(prefix):]
        return ""

    @classmethod
    def combined_conditions(cls, neuron_conditions, vars=[], prefix=0, comments=True):
        """
        Combine the conditions from multiple NeuronCriteria into a single string,
        putting the "cheap" conditions first and the "expensive" conditions last.
        (That is, basic conditions first and the directed ROI conditions last.)
        """
        if isinstance(prefix, int):
            prefix = ' '*prefix

        vars = {*vars}
        for nc in neuron_conditions:
            vars = vars | {nc.matchvar, *nc.global_vars().keys()}
        vars = (*vars,)

        basic_cond = [nc.basic_conditions(0, comments) for nc in neuron_conditions]
        basic_cond = [*filter(None, basic_cond)]
        if not basic_cond:
            return ""

        if basic_cond:
            basic_cond = '\nAND\n'.join(basic_cond)
            basic_cond = indent(basic_cond, ' '*2)
            basic_cond = f"WHERE \n{basic_cond}"

        combined = basic_cond

        roi_conds = [nc.directed_rois_condition(*vars, comments=comments) for nc in neuron_conditions]
        roi_conds = [*filter(None, roi_conds)]
        if roi_conds:
            roi_conds = '\n\n'.join(roi_conds)
            combined = basic_cond + "\n\n" + roi_conds

        return indent(combined, prefix)[len(prefix):]

    def basic_conditions(self, prefix=0, comments=True):
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

        clauses += f"\nAND ".join(exprs)

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


def where_expr(field, values, regex=False, matchvar='n', valuevar=None):
    """
    Return an expression to match a particular
    field against a list of values, to be used
    within the WHERE clause.

    'values' must be a list, and the generated cypher depends on:
        - the length of the list
        - whether or not it contains 'None'
        - whether or not 'regex' is True
        - whether or not 'valuevar' was given

    If 'valuevar' is given, then the generated cypher will refer to the variable
    instead of the literal values, BUT if the literal values contain None,
    then an additional 'exists()' condition will be added.

    Examples:

        .. code-block: ipython

            In [1]: from neuprint.neuroncriteria import where_expr

            In [2]: where_expr('status', [])
            Out[2]: ''

            In [3]: where_expr('status', [None])
            Out[3]: 'NOT exists(n.status)'

            In [4]: where_expr('status', ['Orphan'])
            Out[4]: "n.status = 'Orphan'"

            In [5]: where_expr('status', ['Orphan', 'Assign'])
            Out[5]: "n.status in ['Orphan', 'Assign']"

            In [6]: where_expr('status', ['Orphan', 'Assign', None])
            Out[6]: "n.status in ['Orphan', 'Assign'] OR NOT exists(n.status)"

            In [7]: where_expr('status', ['Orph.*'], regex=True)
            Out[7]: "n.status =~ 'Orph.*'"

            In [8]: where_expr('bodyId', [123])
            Out[8]: 'n.bodyId = 123'

            In [9]: where_expr('bodyId', [123, 456])
            Out[9]: 'n.bodyId in [123, 456]'

            In [10]: where_expr('bodyId', [123, 456, 789], valuevar='bodies')
            Out[10]: 'n.bodyId in bodies'

            In [11]: where_expr('bodyId', [123, None, 456], valuevar='bodies')
            Out[11]: 'n.bodyId in bodies OR NOT exists(n.bodyId)'

            In [12]: where_expr('status', [IsNull])
            Out[12]: 'n.status IS NULL'

            In [13]: where_expr('status', [NotNull])
            Out[13]: 'n.status NOT NULL'
    """
    assert isinstance(values, collections.abc.Iterable), \
        f"Please pass a list or a variable name, not {values}"

    assert valuevar is None or isinstance(valuevar, str)
    assert not regex or not valuevar, "valuevar is not allowed if using a regex"

    if len(values) == 0:
        return ""

    if len(values) == 1:
        if values[0] is None or values[0] == IsNull:
            return f"NOT exists({matchvar}.{field})"

        if values[0] == NotNull:
            return f"exists({matchvar}.{field})"

        if regex:
            return f"{matchvar}.{field} =~ '{values[0]}'"

        if isinstance(values[0], str):
            return f"{matchvar}.{field} = '{values[0]}'"

        return f"{matchvar}.{field} = {values[0]}"

    if NotNull in values and len(values) > 1:
        raise ValueError('`NotNull` can not be combined with other criteria '
                         'for the same field.')

    # list of values
    if None not in values and IsNull not in values:
        if valuevar:
            return f"{matchvar}.{field} in {valuevar}"
        elif regex:
            assert all(isinstance(v, str) for v in values), \
                "Expected all regex values to be strings"
            r = '|'.join(f'({v})' for v in values)
            return f"{matchvar}.{field} =~ '{r}'"
        else:
            return f"{matchvar}.{field} in {[*values]}"

    # ['some_val', None, 'some_other']
    values = [*filter(lambda v: v not in (None, IsNull), values)]
    if len(values) == 1:
        if regex:
            assert isinstance(values[0], str), \
                "Expected all regex values to be strings"
            return f"{matchvar}.{field} =~ '{values[0]}' OR NOT exists({matchvar}.{field})"
        elif isinstance(values[0], str):
            return f"{matchvar}.{field} = '{values[0]}' OR NOT exists({matchvar}.{field})"
        else:
            return f"{matchvar}.{field} = {values[0]} OR NOT exists({matchvar}.{field})"
    else:
        if regex:
            # Combine the list fo regexes into a single regex
            # of the form: '(regex1)|(regex2)|(regex3)'
            assert all(isinstance(v, str) for v in values), \
                "Expected all regex values to be strings"
            r = '|'.join(f'({v})' for v in values)
            return f"{matchvar}.{field} =~ '{r}' OR NOT exists({matchvar}.{field})"
        elif valuevar:
            return f"{matchvar}.{field} in {valuevar} OR NOT exists({matchvar}.{field})"
        else:
            return f"{matchvar}.{field} in {[*values]} OR NOT exists({matchvar}.{field})"
