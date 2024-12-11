Changelog
=========

0.4.26 / 2023-06-08
-------------------
- ``NeuronCriteria`` now supports many new properties for the MANC v1.0 dataset.
- Neuron property columns are determined from cached metadata rather than a full scan of the database.
- If more than one ``Client`` has been constructed, none of them become automatically become the default client.
  In that case, you must explicitly pass a ``client`` argument to each query function you call. This avoids a
  common pitfall when dealing with multiple neuprint datasets (and therefore multiple Clients).
- ``SynapseCriteria`` now uses a default confidence threshold based on the dataset metadata (instead of using 0.0 by default)
- ``Client`` constructor avoids contacting the database unless it needs to. (Duplicate clients are now cheaper to construct.)
- Minor enhancements to skeleton utilities, including a couple new analysis functions.
- Added CITATION.cff

0.4.25 / 2022-09-15
-------------------

- In live-updated neuprint databases, it is possible that an edge's ``weight`` can become out-of-sync with its ``roiInfo`` totals.
  That inconsistency triggered an assertion in ``fetch_adjacencies()``, but now it will emit a warning instead.

0.4.24 / 2022-07-14
-------------------

- Implemented a workaround to avoid a pandas bug in certain cases involving empty dataframes.

0.4.23 / 2022-06-14
-------------------

- In ``fetch_adjacencies()`` (and ``fetch_simple_connections()``), we now ensure that no 0-weight "connections" are returned.

   .. note::

      In recent neuprint databases, some ``:ConnectsTo`` relationships may have a ``weight`` of ``0``.
      In such cases, the relationship will have a non-zero ``weightHR`` (high-recall weight), but all of the relevant
      synapses are low-confidence, hence the "default" ``weight`` of ``0``.
      We now exclude such connections from our results.

0.4.22 / 2022-06-14
-------------------

- Fixed a crash that could occur if you supplied more than three regular expressions for ``type`` or ``instance``.
- Fixed a problem involving 'hidden' ROIs in the hemibrain v1.0.

0.4.21 / 2022-05-14
-------------------

- Now ``heal_skeleton()`` is slightly faster in the case where no healing was necessary.

0.4.20 / 2022-05-13
-------------------

- By default, ``NeuronCriteria`` will now guess whether the ``type`` and ``instance`` contain
  a regular expression or not, so you don't need to explicitly pass ``regex=True``.
  Override the guess by specifying ``regex=True`` or ``regex=False``.

0.4.19 / 2022-05-12
-------------------

- Added ``fetch_mean_synapses()``
- Added ``attach_synapses_to_skeleton()``

0.4.18 / 2022-04-06
-------------------

- Fixed broken package distribution.

0.4.17 / 2022-04-06
-------------------

- **[CHANGE IN RETURNED RESULTS]** ``fetch_synapse_connections()`` now applies ROI filtering criteria to only the post-synaptic points,
  for consistency with ``fetch_adjacencies()``.  (See note in the docs.)
  This means that the number of synapses returned by ``fetch_synapse_connections()`` is now slightly different than it was in previous
  versions of ``neuprint-python``.
- In ``fetch_neurons()``, better handling of old neuprint datasets which lack some fields (e.g. ``upstream``, ``downstream``, ``mito``).

0.4.16 / 2021-11-30
-------------------
- ``NeuronCriteria`` has new fields to support upcoming datasets: ``somaSide``, ``class_``, ``statusLabel``, ``hemilineage``, ``exitNerve``.
- ``NeuronCriteria`` now permits you to search for neurons that contain (or lack) a particular property via a special value ``NotNull`` (or ``IsNull``).
- ``fetch_neurons()`` now returns all neuron properties.
- ``fetch_neurons()`` now returns special rows for NotPrimary connection counts.
- The per-ROI connection counts table returned by ``fetch_neurons()`` now includes rows for connections which fall outside of all primary ROIs.
  These are indicated by the special ROI name ``NotPrimary``.
- ``fetch_synapse_connections()`` uses a more fine-grained batching strategy, splitting the query across more requests to avoid timeouts.
- Fixed a bug in ``fetch_shortest_paths()`` which caused it to generate invalid cypher if the ``intermediate_criteria``
  used a list of bodyIds (or statuses, or rois, etc.) with more than three items.
- ``fetch_output_completeness`` now accepts a list of statuses to use, rather than assuming only ``"Traced"`` neurons are complete.
- Added utility function ``skeleton_segments()``.


0.4.15 / 2021-06-16
-------------------
- ``NeuronCriteria`` now accepts a boolean argument for ``soma``, indicating the presence or absence of a soma on the body.
- Added ``fetch_connection_mitochondria()`` for finding the nearest mitochondria on both sides of a tbar/psd pair. (#24)
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
