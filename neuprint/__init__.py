import os
import platform

from .client import Client, default_client, set_default_client, clear_default_client, list_all_clients
from .queries import ( fetch_custom, fetch_meta, fetch_all_rois, fetch_primary_rois, fetch_roi_hierarchy,
                       fetch_neurons, fetch_custom_neurons, fetch_simple_connections, fetch_adjacencies,
                       fetch_traced_adjacencies, fetch_common_connectivity, fetch_paths, fetch_shortest_paths,
                       fetch_mitochondria, fetch_synapses_and_closest_mitochondria, fetch_connection_mitochondria,
                       fetch_synapses, fetch_mean_synapses, fetch_synapse_connections, fetch_output_completeness,
                       fetch_downstream_orphan_tasks,
                       NeuronCriteria, SegmentCriteria, SynapseCriteria, MitoCriteria )
from .utils import merge_neuron_properties, connection_table_to_matrix, IsNull, NotNull
from .simulation import ( NeuronModel, TimingResult, Ra_LOW, Ra_MED, Ra_HIGH, Rm_LOW, Rm_MED, Rm_HIGH )
from .skeleton import ( fetch_skeleton, skeleton_df_to_nx, skeleton_swc_to_df, skeleton_df_to_swc, heal_skeleton,
                        reorient_skeleton, calc_segment_distances, skeleton_segments, upsample_skeleton,
                        attach_synapses_to_skeleton)
from .wrangle import syndist_matrix, bilateral_syndist, assign_sides_in_groups

from . import _version
__version__ = _version.get_versions()['version']

# On Mac, requests uses a system library which is not fork-safe,
# so using multiprocessing results in segfaults such as the following:
#
#   File ".../lib/python3.7/urllib/request.py", line 2588 in proxy_bypass_macosx_sysconf
#   File ".../lib/python3.7/urllib/request.py", line 2612 in proxy_bypass
#   File ".../lib/python3.7/site-packages/requests/utils.py", line 745 in should_bypass_proxies
#   File ".../lib/python3.7/site-packages/requests/utils.py", line 761 in get_environ_proxies
#   File ".../lib/python3.7/site-packages/requests/sessions.py", line 700 in merge_environment_settings
#   File ".../lib/python3.7/site-packages/requests/sessions.py", line 524 in request
#   File ".../lib/python3.7/site-packages/requests/sessions.py", line 546 in get
# ...

# The workaround is to set a special environment variable
# to avoid the particular system function in question.
# Details here:
# https://bugs.python.org/issue30385
if platform.system() == "Darwin":
    os.environ["no_proxy"] = "*"
