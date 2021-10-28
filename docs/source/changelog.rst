Changelog
=========

0.4.15 / 2021-06-16
-------------------
- ``NeuronCriteria`` now accepts a boolean argument for ``soma``, indicating the presence or absence of a soma on the body.
- Added ``fech_connection_mitochondria()`` for finding the nearest mitochondria on both sides of a tbar/psd pair. (#24)
- Integrated with Zenodo for DOI generation.

0.4.14 / 2021-03-27
-------------------
- Updated to changes in the neuPrint mitochondria data model.
  Older versions of ``neuprint-python`` cannot query for mitochondria any more.
- ``fetch_neurons()``: Added new columns to the ``roi_counts_df`` result, for ``upstream, downstream, mito``
- ``fetch_skeletons()``: Now supports ``with_distances`` option
- ``NeuronCriteria`` permits lists of strings for type/instance regular expressions.
  (Previously, lists were only permitted when ``regex=False``.)
- Fixed a performance problem in ``fetch_synapse_connections()``
- More FAQ entries


0.4.13 / 2020-12-23
-------------------

- ``SynapseCriteria``: Changed the default value of ``primary_only`` to ``True``,
  since it may been counter-intuitive to obtain duplicate results by default.
- ``NeuronCriteria``: Added ``cellBodyFiber`` parameter. (Philipp Shlegel #13)
- Added mitochondria queries


0.4.12 / 2020-11-21
-------------------

- Better handling when adjacency queries return empty results
- Simulation: Minor change to subprocess communication implementation
- Skeleton DataFrames use economical dtypes
- Minor bug fixes and performance enhancements
- fetch_synapse_connections(): Fix pandas error in assertion


0.4.11 / 2020-06-30
-------------------

- Fixed ``ngspice`` install instructions.


0.4.10 / 2020-06-30
-------------------

- Moved skeleton-related functions into their own module, and added a few more skeleton utilty functions
- Simulation: Support Windows
- ``heal_skeleton():`` Allow caller to specify a maximum distance for repaired skeleton segments (#12)


0.4.9 / 2020-04-29
------------------

- Added simulation functions and tutorial
