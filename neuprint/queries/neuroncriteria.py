import os
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

from ..utils import ensure_list_args, ensure_list_attrs, IsNull, NotNull, cypher_identifier
from ..client import inject_client

NoneType = type(None)

def neuroncriteria_args(*argnames):
    """
    Returns a decorator.
    For the given argument names, the decorator converts the
    arguments into NeuronCriteria objects via ``copy_as_neuroncriteria()``.

    If provided, the 'client' keyword argument to the decorated function
    must be the same as the 'client' attribute in all of the passed NeuronCriteria arguments.
    If any client is None (either the keyword argument or the NeuronCriteria client(s)),
    then it will be automatically set from the valid clients that were provided.

    Example:

    .. code-block:: python

        @neuroncriteria_args('nc'):
        def foo(nc, client=None):
            pass

        # All of the following are equivalent:
        foo(NC(client=c), client=c)
        foo(NC(client=c))  
        foo(NC(), client=c)

        # Invalid:
        c1 = Client('neuprint.janelia.org', 'hemibrain:v1.0')
        c2 = Client('neuprint.janelia.org', 'hemibrain:v1.1')
        foo(NC(client=c1), client=c2)  # raises ValueError
    """
    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            callargs = inspect.getcallargs(f, *args, **kwargs)

            clients = set()
            if c := callargs.get('client', None):
                clients.add(c)
            for name in argnames:
                arg = callargs[name]
                if isinstance(arg, NeuronCriteria) and arg.client:
                    clients.add(arg.client)

            if len(clients) == 1:
                client = clients.pop()
            elif len(clients) > 1:
                raise ValueError(f"Clients of NeuronCriteria arguments and/or the keyword argument do not match: {clients}")
            else:
                client = None

            for name in argnames:
                callargs[name] = copy_as_neuroncriteria(callargs[name], client)

            if 'client' in callargs:
                callargs['client'] = client

            return f(**callargs)

        wrapper.__signature__ = inspect.signature(f)
        return wrapper

    return decorator


def copy_as_neuroncriteria(obj, client=None):
    """
    If the given argument is a NeuronCriteria object, copy it
    and overwrite the copy's 'client' attribute if one was provided.

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
        nc = copy.copy(obj)
        if client:
            nc.client = client
        return nc

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


# We permit the user to provide these NeuronCriteria members as
# single values, lists, arrays, Series, etc. (or None).
# But to facilitate a simpler implementation, we automatically
# convert them all to lists upon class initialization and/or attribute
# setting (via decorators).
_iterable_attrs = [
    'bodyId',

    # Regex-optional
    'type', 'instance',
    # special: 'regex'

    # ROI
    'rois', 'inputRois', 'outputRois',
    # special: roi_req, min_roi_inputs, min_roi_outputs

    # integer
    'group', 'serial', 'zapbenchId',

    # boolean
    # 'cropped',

    # Secondary type fields (regex-optional)
    'flywireType', 'hemibrainType', 'mancType',

    # Exact string fields (alphabetical order)
    # these fields come from several datasets; this list appears elsewhere
    #   in the code, always with "exact string fields" in a nearby comment
    # TODO: (a) generate this list dynamically from the db, and
    #       (b) consider making them all eligible for regex searching
    'birthtime', 'cellBodyFiber', 'celltypePredictedNt',
    'class_', 'consensusNt', 'description',
    'dimorphism', 'entryNerve', 'exitNerve', 'flywireId',
    'fruDsx', 'hemibrainBodyId',
    'hemilineage', 'itoleeHl', 'locationType', 'longTract',
    'matchingNotes', 'mcnsSerial', 'modality',
    'ntReference', 'origin', 'otherNt', 'otherNtReference',
    'predictedNt', 'prefix', 'receptorType', 'rootSide',
    'serialMotif', 'somaNeuromere', 'somaSide', 'source',
    'status', 'statusLabel', 'subclass', 'subclassabbr',
    'superclass', 'supertype', 'synonyms', 'systematicType',
    'tag', 'target', 'transmission', 'trumanHl',
    'vfbId',

    # Special
    # label, min_pre, min_post, max_closestLandmarkDistanceMicrons

    # Null/NotNull
    # somaLocation, tosomaLocation,

    # Deprecated
    # soma
]


@ensure_list_attrs(_iterable_attrs)
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

    @ensure_list_args(_iterable_attrs)
    def __init__(
        self, matchvar='n', *,
        bodyId=None,

        # Regex-optional
        type=None, instance=None,
        regex='guess',

        # status (exact string)
        status=None, statusLabel=None,

        # ROI
        rois=None, inputRois=None, outputRois=None,
        roi_req='all', min_roi_inputs=1, min_roi_outputs=1,

        # integer
        group=None, serial=None, zapbenchId=None,

        # boolean
        cropped=None,

        # Secondary type fields (regex-optional)
        flywireType=None, hemibrainType=None, mancType=None,

        # Other exact string fields (alphabetical)
        birthtime=None, cellBodyFiber=None, celltypePredictedNt=None,
        class_=None, consensusNt=None, description=None,
        dimorphism=None, entryNerve=None, exitNerve=None, flywireId=None,
        fruDsx=None, hemibrainBodyId=None,
        hemilineage=None, itoleeHl=None, locationType=None, longTract=None,
        matchingNotes=None, mcnsSerial=None, modality=None,
        ntReference=None, origin=None, otherNt=None, otherNtReference=None,
        predictedNt=None, prefix=None, receptorType=None, rootSide=None,
        serialMotif=None, somaNeuromere=None, somaSide=None, source=None,
        subclass=None, subclassabbr=None,
        superclass=None, supertype=None, synonyms=None, systematicType=None,
        tag=None, target=None, transmission=None, trumanHl=None,
        vfbId=None,


        # Special
        label=None, min_pre=0, min_post=0, max_closestLandmarkDistanceMicrons=np.inf,

        # IsNull/NotNull
        somaLocation=None, tosomaLocation=None, rootLocation=None,

        # Deprecated
        soma=None,

        client=None
    ):
        """
        All criteria must be passed as keyword arguments.

        .. note::
            Only ``bodyId``, ``type``, ``instance``, and ROI-related criteria are
            applicable to all datasets.  The applicability of all other criteria depends
            on the dataset being accessed (e.g. hemibrain,  manc, etc.).

        .. note::

            **Options for specifying ROI criteria**

            The ``rois`` argument merely matches neurons that intersect the given ROIs at all
            (without distinguishing between inputs and outputs).

            The ``inputRois`` and ``outputRois`` arguments allow you to put requirements
            on whether or not neurons have inputs or outputs in the listed ROIs.
            It produces a more expensive query, but it's more selective.
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
                Must begin with a lowercase letter.

            bodyId (int or list of ints):
                List of bodyId values.

            type (str or list of str):
                Cell type.  Matches depend on the the ``regex`` argument.
                If both ``type`` and ``instance`` criteria are supplied, any neuron that
                matches EITHER criteria will match the overall criteria.

            instance (str or list of str):
                Cell instance (specific cell name).  Matches depend on the the ``regex`` argument.
                If both ``type`` and ``instance`` criteria are supplied, any neuron that
                matches EITHER criteria will match the overall criteria.

            regex (bool):
                If ``True``, the ``instance`` and all ``type``-style arguments will be interpreted
                as regular expressions, rather than exact match strings.
                If ``False``, only exact matches will be found.
                By default, the matching method will be automatically chosen by inspecting the
                ``type`` and ``instance`` strings.  If they contain regex-like characters,
                then we assume you intend regex matching. (You can see which method was chosen by
                checking the ``regex`` field after the ``NeuronCriteria`` is constructed.)

            status (str or list of str):
                Indicates the status of the neuron's reconstruction quality.
                Typically, named/annotated neurons have ``Traced`` status,
                the best quality.
            statusLabel (str or list of str):
               ``statusLabel`` is typically more fine-grained than ``status``,
               and mostly of interest during the construction of the connectome,
               not for end-users.  The possible values of ``statusLabel`` do not
               correspond one-to-one to those of ``status``.

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

            group (int or list of int)
                In some datasets, the ``group`` ID is used to associate neurons morphological type,
                including left-right homologues. Neurons with the same group ID have matching morphology.

            serial (int or list of int)
                Similar to ``group``, but used for associating neurons across segmental neuropils in the nerve cord.
                Neurons with the same ``serial`` ID are analogous to one another, but in different leg segments.

            cropped (bool):
                If given, restrict results to neurons that are cropped or not.

            dataset-specific exact string fields, possibiliy including but not limited to:
                birthtime (str or list of str):
                cellBodyFiber (str or list of str):
                superclass (str or list of str):
                class\\_ (str or list of str):
                    Matches for the neuron ``class`` field.
                entryNerve (str or list of str):
                exitNerve (str or list of str):
                flywireType (str or list of str, regex-optional):
                hemibrainType (str or list of str, regex-optional):
                hemilineage (str or list of str):
                itoleeHl (str or list of str):
                longTract (str or list of str):
                mancType (str or list of str, regex-optional):
                modality (str or list of str):
                origin (str or list of str):
                predictedNt (str or list of str):
                serialMotif (str or list of str):
                somaNeuromere (str or list of str):
                somaSide  (str or list of str):
                    Valid choices are 'RHS', 'LHS', 'Midline'
                subclass (str or list of str):
                synonyms (str or list of str):
                systematicType (str or list of str):
                target (str or list of str):
                trumanHl (str or list of str):
                zapbenchId

            label (Either ``'Neuron'`` or ``'Segment'``):
                Which node label to match with.
                (In neuprint, all ``Neuron`` nodes are also ``Segment`` nodes.)
                By default, ``'Neuron'`` is used, unless you provided a non-empty ``bodyId`` list.
                In that case, ``'Segment'`` is the default. (It's assumed you're really interested
                in the bodies you explicitly listed, whether or not they have the ``'Neuron'`` label.)

            min_pre (int):
                Exclude neurons that don't have at least this many t-bars (outputs) overall,
                regardless of how many t-bars exist in any particular ROI.

            min_post (int):
                Exclude neurons that don't have at least this many PSDs (inputs) overall,
                regardless of how many PSDs exist in any particular ROI.

            somaLocation:
                The ``somaLocation`` property of ``:Neuron`` objects contains
                the ``[X,Y,Z]`` coordinate (in voxels) of the cell body.
                ``NeuronCriteria`` does not allow you to match a specific coordinate,
                but you may set this argument to ``NotNull` (or ```IsNull``) to
                search for cells with (or without) a recorded cell body.

            tosomaLocation:
                Neurons which could not be successfully attached to their cell body do not have
                a recorded ``somaLocation``.  Instead, they have an annotaiton on the cell body
                fiber, on the severed end extending out toward the cell body.
                Like ``somaLocation``, you can't match a specific coordinate using ``NeuronCriteria``,
                but you can use ``NotNull``/``IsNull``.

            rootLocation:
                Some (but not all) Neurons which have no soma in the tissue sample are tagged with
                a ``rootLocation``, indicating where they enter/exit the sample.
                Like ``somaLocation``, you can't match a specific coordinate using ``NeuronCriteria``,
                but you can use ``NotNull``/``IsNull``.

            soma (Either ``True``, ``False``, or ``None``)
                DEPRECATED.  Use ``somaLocation=NotNull`` or ``somaLocation=IsNull``.

            zapbenchId (int or list of int):
                (zebrafish only) This field is used to identify neurons that have been matched to neurons
                in the ZAPBench dataset.

            max_closestLandmarkDistanceMicrons (float):
                (zebrafish only) The maximum distance (in microns) from the closest landmark
                that a neuron must have to be included in the results.

            client (:py:class:`neuprint.client.Client`):
                Used to validate ROI names.
        """
        # Note: client may be None upon construction, but typically
        # the @neuroncriteria_args decorator will set it automatically.
        self.client = client

        self.matchvar = self._init_matchvar(matchvar)
        self.bodyId = self._init_integer_arg(bodyId, 'bodyId')

        # regex-optional
        self.type = self._init_type(type)
        self.instance = self._init_instance(instance)

        # Status (exact string)
        self.status = status
        self.statusLabel = statusLabel

        # ROI
        (self.roi_req, self.min_roi_inputs, self.min_roi_outputs,
         self.rois, self.inputRois, self.outputRois) = (
            self._init_rois(
                roi_req, min_roi_inputs, min_roi_outputs,
                rois, inputRois, outputRois
            )
        )

        # integer
        self.group = self._init_integer_arg(group, 'group')
        self.serial = self._init_integer_arg(serial, 'serial')
        self.zapbenchId = self._init_integer_arg(zapbenchId, 'zapbenchId')

        # boolean
        self.cropped = cropped

        # NotNull/IsNull
        self.somaLocation = self._init_location_arg(somaLocation, 'somaLocation')
        self.tosomaLocation = self._init_location_arg(tosomaLocation, 'tosomaLocation')
        self.rootLocation = self._init_location_arg(rootLocation, 'rootLocation')

        # Alternative type fields (alphabetical)
        self.flywireType = flywireType
        self.hemibrainType = hemibrainType
        self.mancType = mancType

        # Other exact string fields (alphabetical order)
        self.birthtime = birthtime
        self.cellBodyFiber = cellBodyFiber
        self.celltypePredictedNt = celltypePredictedNt
        self.class_ = class_
        self.consensusNt = consensusNt
        self.description = description
        self.dimorphism = dimorphism
        self.entryNerve = entryNerve
        self.exitNerve = exitNerve
        self.flywireId = flywireId
        self.fruDsx = fruDsx
        self.hemibrainBodyId = hemibrainBodyId
        self.hemilineage = hemilineage
        self.itoleeHl = itoleeHl
        self.locationType = locationType
        self.longTract = longTract
        self.matchingNotes = matchingNotes
        self.mcnsSerial = mcnsSerial
        self.modality = modality
        self.ntReference = ntReference
        self.origin = origin
        self.otherNt = otherNt
        self.otherNtReference = otherNtReference
        self.predictedNt = predictedNt
        self.prefix = prefix
        self.receptorType = receptorType
        self.rootSide = rootSide
        self.serialMotif = serialMotif
        self.somaNeuromere = somaNeuromere
        self.somaSide = somaSide
        self.source = source
        self.subclass = subclass
        self.subclassabbr = subclassabbr
        self.superclass = superclass
        self.supertype = supertype
        self.synonyms = synonyms
        self.systematicType = systematicType
        self.tag = tag
        self.target = target
        self.transmission = transmission
        self.trumanHl = trumanHl
        self.vfbId = vfbId

        # Special
        self.label = self._init_label(label, bodyId)
        self.min_pre = min_pre
        self.min_post = min_post
        self.max_closestLandmarkDistanceMicrons = max_closestLandmarkDistanceMicrons

        # Deprecated
        self.soma = self._init_soma(soma, somaLocation)

        # These are the properties for which we can encode the possible values
        # in a cypher list literal and use syntax like "WHERE prop in prop_list".
        # Does not include ROIs (since they are stored as multiple boolean properties).
        self.list_props = [
            'bodyId',

            # integer
            'group', 'serial', 'zapbenchId',

            # status (exact string)
            'status', 'statusLabel',

            # Other exact string fields (alphabetical order)
            'birthtime', 'cellBodyFiber', 'celltypePredictedNt',
            'class_', 'consensusNt', 'description',
            'dimorphism', 'entryNerve', 'exitNerve', 'flywireId',
            'flywireType', 'fruDsx', 'hemibrainBodyId', 'hemibrainType',
            'hemilineage', 'itoleeHl', 'locationType', 'longTract',
            'mancType', 'matchingNotes', 'mcnsSerial', 'modality',
            'ntReference', 'origin', 'otherNt', 'otherNtReference',
            'predictedNt', 'prefix', 'receptorType', 'rootSide',
            'serialMotif', 'somaNeuromere', 'somaSide', 'source',
            'subclass', 'subclassabbr',
            'superclass', 'supertype', 'synonyms', 'systematicType',
            'tag', 'target', 'transmission', 'trumanHl',
            'vfbId',

        ]
        self.list_props_regex = ['type', 'instance', 'flywireType', 'hemibrainType', 'mancType']
        self.regex = self._init_regex(regex, *[getattr(self, prop) for prop in self.list_props_regex])

    @classmethod
    def _init_matchvar(cls, matchvar):
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
        return matchvar

    @classmethod
    def _init_integer_arg(cls, arg, name):
        assert len(arg) == 0 or np.issubdtype(np.asarray(arg).dtype, np.integer), \
            f"{name} should be an integer or list of integers"
        return arg

    @classmethod
    def _init_location_arg(cls, arg, name):
        assert arg in (IsNull, NotNull, None), \
            (f"This function doesn't allow you to search for an exact {name}.\n"
             f"You can only check for the presence or absence of the {name} property via IsNull or NotNull.\n"
             f"If you need to search for a particular cell via th {name} property, write a custom Cypher query.")
        return arg

    @classmethod
    def _init_label(cls, label, bodyId):
        if not label:
            if len(bodyId) == 0:
                label = 'Neuron'
            else:
                label = 'Segment'
        assert label in ('Neuron', 'Segment'), f"Invalid label: {label}"
        return label

    @classmethod
    def _init_type(cls, type):
        for t in type:
            assert isinstance(t, (str, NoneType)) or t in (IsNull, NotNull), \
                f'type should be a string, IsNull, NotNull or None, got {t}'
        return type

    @classmethod
    def _init_instance(cls, instance):
        for i in instance:
            assert isinstance(i, (str, NoneType)) or i in (IsNull, NotNull), \
                f'instance should be a string, IsNull, NotNull or None, got {i}'
        return instance

    @classmethod
    def _init_regex(cls, regex, *args):
        assert regex in (True, False, 'guess')
        if regex != 'guess':
            return regex

        rgx = re.compile(r'[\\\.\?\[\]\+\^\$\*]')
        for arg in args:
            for val in arg:
                if isinstance(val, str) and bool(rgx.search(val or '')):
                    return True

        return False

    @classmethod
    def _init_rois(cls, roi_req, min_roi_inputs, min_roi_outputs, rois, inputRois, outputRois):
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
        assert not rois or rois >= inputRois, "Queried intersecting rois must be a superset of the inputRois"
        assert not rois or rois >= outputRois, "Queried intersecting rois must be a superset of the outputRois"

        # Make sure intersecting is a superset of inputRois and outputRois
        rois |= {*inputRois, *outputRois}

        return roi_req, min_roi_inputs, min_roi_outputs, rois, inputRois, outputRois

    @inject_client
    def _validate_rois(self, *, client=None):
        """
        Verify ROI names against known ROIs.
        This requires access to a client from which we obtain the list of valid ROIs.
        We don't call this function upon construction, since sometimes it's
        convenient to assign the client after the rest of the initialization.
        Instead, this function is called when constructing a query.
        """

        neuprint_rois = {*client.all_rois}
        unknown_input_rois = self.inputRois - neuprint_rois
        if unknown_input_rois:
            raise RuntimeError(f"Unrecognized input ROIs: {unknown_input_rois}")

        unknown_output_rois = self.outputRois - neuprint_rois
        if unknown_output_rois:
            raise RuntimeError(f"Unrecognized output ROIs: {unknown_output_rois}")

        unknown_generic_rois = self.rois - neuprint_rois
        if unknown_generic_rois:
            raise RuntimeError(f"Unrecognized ROIs: {unknown_generic_rois}")

    @classmethod
    def _init_soma(cls, soma, somaLocation):
        assert soma in (True, False, None), \
            f"soma must be True, False or None, not {soma}"
        assert (soma is None) or (somaLocation is not None), \
            ("Can't use both 'soma' and 'somaLocation' arguments; "
             "Since 'soma' is deprecated, just use e.g. somaLocation=NotNull")
        return soma

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
        # (DO include the client)
        params = inspect.signature(NeuronCriteria).parameters.keys()
        params = {*params} - {'matchvar'}

        for p in params:
            me = getattr(self, p)
            other = getattr(value, p)

            # If not the same type, return False
            if type(me) is not type(other):
                return False

            # If iterable (e.g. ROIs or body IDs) we don't care about order
            if isinstance(me, Iterable):
                if set(me) != set(other):
                    return False
            elif me != other:
                return False

        # If all comparisons have passed, return True
        return True

    def __repr__(self):
        # Show all non-default constructor args
        s = f'NeuronCriteria("{self.matchvar}"'

        list_props = [self.list_props[0], *self.list_props_regex, *self.list_props[1:]]

        if self.label != 'Neuron':
            s += f', label="{self.label}"'

        for attr in list_props:
            val = getattr(self, attr)
            if len(val) == 1:
                s += f', {attr}={repr(val[0])}'
            elif len(self.instance) > 1:
                s += f", {attr}={repr(list(val))}"

        if any(
            len(getattr(self, attr))
            for attr in self._list_props_regex
        ):
            s += f", regex={self.regex}"

        if self.min_pre != 0:
            s += f", min_pre={self.min_pre}"

        if self.min_post != 0:
            s += f", min_post={self.min_post}"

        if not np.isposinf(self.max_closestLandmarkDistanceMicrons):
            s += f", max_closestLandmarkDistanceMicrons={self.max_closestLandmarkDistanceMicrons}"

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

        if self.roi_req != 'all':
            s += f', roi_req="{self.roi_req}"'

        for attr in ['cropped', 'somaLocation', 'tosomaLocation', 'rootLocation', 'soma']:
            val = getattr(self, attr)
            if val is not None:
                s += f", {attr}={val}"

        s += ')'

        return s

    MAX_LITERAL_LENGTH = 3
    assert MAX_LITERAL_LENGTH >= 3, \
        ("The logic in where_expr() assumes valuevars "
         "have length 3 (assuming one could be None).")

    def global_vars(self):
        exprs = {}

        if self.regex:
            list_props = self.list_props
        else:
            # No regex: Treat instance and type as ordinary strings
            list_props = self.list_props + self.list_props_regex

        for key in list_props:
            values = getattr(self, key)
            if len(values) > self.MAX_LITERAL_LENGTH:
                if key.endswith('_'):
                    key = key[:-1]
                if isinstance(values, np.ndarray):
                    values = values.tolist()
                values = [v for v in values if v is not None]
                var = cypher_identifier(f"{self.matchvar}_search_{key}")
                exprs[var] = (f"{repr(values)} as {var}")

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
        # Most expressions are simple exact value matches.
        exprs = {}
        for prop in self.list_props:
            val = getattr(self, prop)
            if prop.endswith('_'):
                key = prop[:-1]
            else:
                key = prop
            expr = self._value_list_expr(key, val, False)
            exprs[(prop,)] = expr

        # These are other types of expressions.
        exprs |= {
            ('type', 'instance', 'regex'): self.typeinst_expr(),
            ('cropped',): self._tag_expr('cropped', self.cropped),
            ('somaLocation',): self._nullcheck_expr('somaLocation', self.somaLocation),
            ('tosomaLocation',): self._nullcheck_expr('tosomaLocation', self.tosomaLocation),
            ('rootLocation',): self._nullcheck_expr('rootLocation', self.rootLocation),
            ('soma',): self._single_value_expr('somaLocation', self.soma),  # deprecated arg
            ('min_pre',): self._gt_eq_expr('pre', self.min_pre),
            ('min_post',): self._gt_eq_expr('post', self.min_post),
            ('max_closestLandmarkDistanceMicrons',): self._lt_eq_expr('closestLandmarkDistanceMicrons',
                  self.max_closestLandmarkDistanceMicrons),
            ('rois', 'inputRois', 'outputRois', 'roi_req', 'min_roi_inputs', 'min_roi_outputs'): self.rois_expr(),
            # No expression for label;
            # enclosing queries are responsible for inserting label into their MATCH statement.
            ('label',): "",
        }

        # Since we've got a lot of criteria to generate expressions for,
        # let's verify that we remembered to implement expressions for every
        # argument in the NeuronCriteria constructor.
        if 'PYTEST_CURRENT_TEST' in os.environ:
            sig = inspect.signature(NeuronCriteria)
            criteria_args = set(sig.parameters.keys()) - {'matchvar', 'client'}
            missing_exprs = criteria_args - set(chain(*exprs.keys()))
            assert not missing_exprs, \
                ("NeuronCriteria.basic_exprs() doesn't have a Cypher expression "
                f"for all criteria!  Missing: {missing_exprs}'")

        return [*filter(None, exprs.values())]

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
            return f"{self.matchvar}.{key} = {repr(value)}"
        elif value:
            return f"{self.matchvar}.{key} IS NOT NULL"
        else:
            return f"{self.matchvar}.{key} IS NULL"

    def _nullcheck_expr(self, key, value):
        assert value in (None, IsNull, NotNull)
        if value == IsNull:
            return f"NOT exists({self.matchvar}.{key})"
        elif value == NotNull:
            return f"exists({self.matchvar}.{key})"
        else:
            return ""

    def _tag_expr(self, key, value):
        """
        Match against tag, e.g. `.cropped`.
        Non-existing tags are counted as False.
        """
        if value is None:
            return ""

        key = cypher_identifier(key)

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
        tag_properties = (f"{self.matchvar}.{cypher_identifier(v)}" for v in tags)
        return "(" + f" {logic} ".join(tag_properties) + ")"

    def _gt_eq_expr(self, key, value):
        """
        Match against key/value being greater or equal.
        """
        if value:
            return f"{self.matchvar}.{key} >= {value}"
        else:
            return ""

    def _lt_eq_expr(self, key, value):
        """
        Match against key/value being less than or equal.
        """
        if not np.isposinf(value):
            return f"{self.matchvar}.{key} <= {value}"
        else:
            return ""

    def typeinst_expr(self):
        """
        Unlike all other fields, type and instance OR'd together.
        Either match satisfies the criteria.
        """
        t = self._value_list_expr('type', self.type, self.regex)
        i = self._value_list_expr('instance', self.instance, self.regex)

        if t and i:
            return f"({t} OR {i})"
        if t:
            return t
        if i:
            return i
        return ""

    def rois_expr(self):
        self._validate_rois(client=self.client)
        return self._logic_tag_expr(
            self.rois,
            {'any': 'OR', 'all': 'AND'}[self.roi_req]
        )

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

        clauses += "\nAND ".join(exprs)

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
                 {repr([*self.inputRois])} as inputRois,
                 {repr([*self.outputRois])} as outputRois,
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
        # RETURN n, matchingInputRois, matchingOutputRois

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

            In [8]: where_expr('instance', ['foo.*', 'bar.*', 'baz.*'], regex=True)
            Out[8]: "n.instance =~ '(foo.*)|(bar.*)|(baz.*)'"

            In [9]: where_expr('bodyId', [123])
            Out[9]: 'n.bodyId = 123'

            In [10]: where_expr('bodyId', [123, 456])
            Out[10]: 'n.bodyId in [123, 456]'

            In [11]: where_expr('bodyId', [123, 456, 789], valuevar='bodies')
            Out[11]: 'n.bodyId in bodies'

            In [12]: where_expr('bodyId', [123, None, 456], valuevar='bodies')
            Out[12]: 'n.bodyId in bodies OR NOT exists(n.bodyId)'

            In [13]: where_expr('status', [IsNull])
            Out[13]: 'n.status IS NULL'

            In [14]: where_expr('status', [NotNull])
            Out[14]: 'n.status NOT NULL'
    """
    assert isinstance(values, collections.abc.Iterable), \
        f"Please pass a list or a variable name, not {values}"

    if isinstance(values, np.ndarray):
        values = values.tolist()
    else:
        values = list(values)

    assert valuevar is None or isinstance(valuevar, str)
    assert not regex or not valuevar, "valuevar is not allowed if using a regex"

    field = cypher_identifier(field)

    if len(values) == 0:
        return ""

    if len(values) == 1:
        if values[0] is None or values[0] == IsNull:
            return f"NOT exists({matchvar}.{field})"

        if values[0] == NotNull:
            return f"exists({matchvar}.{field})"

        if regex:
            return f"{matchvar}.{field} =~ {repr(values[0])}"

        return f"{matchvar}.{field} = {repr(values[0])}"

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
            return f"{matchvar}.{field} =~ {repr(r)}"
        else:
            return f"{matchvar}.{field} in {repr(values)}"

    # ['some_val', None, 'some_other']
    values = [v for v in values if v not in (None, IsNull)]
    if len(values) == 1:
        if regex:
            assert isinstance(values[0], str), \
                "Expected all regex values to be strings"
            return f"{matchvar}.{field} =~ {repr(values[0])} OR NOT exists({matchvar}.{field})"
        else:
            return f"{matchvar}.{field} = {repr(values[0])} OR NOT exists({matchvar}.{field})"
    else:
        if regex:
            # Combine the list of regexes into a single regex
            # of the form: '(regex1)|(regex2)|(regex3)'
            assert all(isinstance(v, str) for v in values), \
                "Expected all regex values to be strings"
            r = '|'.join(f'({v})' for v in values)
            return f"{matchvar}.{field} =~ {repr(r)} OR NOT exists({matchvar}.{field})"
        elif valuevar:
            return f"{matchvar}.{field} in {valuevar} OR NOT exists({matchvar}.{field})"
        else:
            return f"{matchvar}.{field} in {repr(values)} OR NOT exists({matchvar}.{field})"
