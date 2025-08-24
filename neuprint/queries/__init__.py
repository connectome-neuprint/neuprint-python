from .neuroncriteria import NeuronCriteria, SegmentCriteria
from .mitocriteria import MitoCriteria
from .synapsecriteria import SynapseCriteria
from .general import fetch_custom, fetch_meta
from .rois import fetch_all_rois, fetch_primary_rois,fetch_roi_hierarchy
from .neurons import fetch_neurons, fetch_custom_neurons
from .connectivity import (fetch_simple_connections, fetch_adjacencies, fetch_traced_adjacencies,
                           fetch_common_connectivity, fetch_paths, fetch_shortest_paths)
from .synapses import fetch_synapses, fetch_mean_synapses, fetch_synapse_connections
from .mito import fetch_mitochondria, fetch_synapses_and_closest_mitochondria, fetch_connection_mitochondria
from .recon import fetch_output_completeness, fetch_downstream_orphan_tasks

# Change the __module__ of each function to make it look like it was defined in this file.
# This hackery is to make the Sphinx autosummary and autodoc play nicely together.
for k, v in dict(locals()).items():
    if k.startswith('fetch'):
        v.__module__ = 'neuprint.queries'
del k, v
