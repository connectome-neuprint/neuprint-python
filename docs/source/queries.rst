.. currentmodule:: neuprint.queries
  

..
   (The following |br| definition is the only way
   I can force numpydoc to display explicit newlines...) 

.. |br| raw:: html

   <br />


.. _queries:

Common Queries
==============

Convenience functions for common queries.

If you are familiar with the neuPrint data model and the
`cypher <https://neo4j.com/developer/cypher-query-language/>`_
query language, you can write your own queries using
:py:func:`fetch_custom <fetch_custom>`.
But the functions in this file offer a convenient API for common queries.

For neuprint's "built-in" (non-cypher) queries, such as skeletons, ROI meshes,
ROI connectivity, and server metadata, see the :py:class:`.Client` class reference.

General
-------

.. autosummary::

    fetch_custom
    fetch_meta
    fetch_all_rois
    fetch_primary_rois

Neurons
-------

.. autosummary::

    fetch_neurons
    fetch_custom_neurons

Connectivity
------------

.. autosummary::

    fetch_simple_connections
    fetch_adjacencies
    fetch_traced_adjacencies
    fetch_common_connectivity
    fetch_shortest_paths

Synapses
--------

.. autosummary::

    fetch_synapses
    fetch_synapse_connections


.. automodule:: neuprint.queries
   :members:

