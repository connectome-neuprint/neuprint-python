.. currentmodule:: neuprint.queries


..
   (The following |br| definition is the only way
   I can force numpydoc to display explicit newlines...)

.. |br| raw:: html

   <br />


.. _queries:

==============
Common Queries
==============

Convenience functions for common queries.

If you are familiar with the neuPrint data model and the
`cypher <https://neo4j.com/developer/cypher-query-language/>`_
query language, you can write your own queries using
:py:func:`fetch_custom <fetch_custom>`.
But the functions in this file offer a convenient API for common queries.

Server Built-in Queries
=======================


See the :ref:`Client` class reference for the neuprint server's built-in
(non-cypher) queries, such as **skeletons**, **ROI meshes**, **ROI connectivity**,
and server metadata.

.. note::

    The functions below are all available via the ``neuprint`` top-level namespace.
    You need not pay attention to the fine-grained module names below.

    .. code-block:: python

        from neuprint import fetch_neurons, fetch_adjacencies, NeuronCriteria as NC

General
=======

.. autosummary::

    fetch_custom
    fetch_meta

ROIs
====

.. autosummary::

    fetch_all_rois
    fetch_primary_rois
    fetch_roi_hierarchy

.. seealso::

    - :py:meth:`.Client.fetch_roi_completeness()`
    - :py:meth:`.Client.fetch_roi_connectivity()`

Neurons
=======

.. autosummary::

    fetch_neurons
    fetch_custom_neurons

Connectivity
============

.. autosummary::

    fetch_simple_connections
    fetch_adjacencies
    fetch_traced_adjacencies
    fetch_common_connectivity
    fetch_shortest_paths

Synapses
========

.. autosummary::

    fetch_synapses
    fetch_synapse_connections

Mitochondria
============

.. autosummary::

    fetch_mitochondria
    fetch_synapses_and_closest_mitochondria
    fetch_connection_mitochondria

Reconstruction Tools
====================

.. autosummary::

    fetch_output_completeness
    fetch_downstream_orphan_tasks


Reference
=========

.. I can't figure out how to make automodule display these in the 'bysource' order, so I'm specifying the order explicitly.

General
-------

.. autofunction:: fetch_custom
.. autofunction:: fetch_meta

ROIs
----

.. autofunction:: fetch_all_rois
.. autofunction:: fetch_primary_rois
.. autofunction:: fetch_roi_hierarchy

Neurons
-------

.. autofunction:: fetch_neurons
.. autofunction:: fetch_custom_neurons

Connectivity
------------

.. autofunction:: fetch_simple_connections
.. autofunction:: fetch_adjacencies
.. autofunction:: fetch_traced_adjacencies
.. autofunction:: fetch_common_connectivity
.. autofunction:: fetch_shortest_paths

Synapses
--------

.. autofunction:: fetch_synapses
.. autofunction:: fetch_synapse_connections

Mitochondria
------------

.. autofunction:: fetch_mitochondria
.. autofunction:: fetch_synapses_and_closest_mitochondria
.. autofunction:: fetch_connection_mitochondria

Reconstruction Tools
--------------------

.. autofunction:: fetch_output_completeness
.. autofunction:: fetch_downstream_orphan_tasks
