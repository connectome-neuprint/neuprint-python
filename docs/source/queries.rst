.. currentmodule:: neuprint.queries
  
.. _queries:


..
   (The following |br| definition is the only way
   I can force numpydoc to display explicit newlines...) 

.. |br| raw:: html

   <br />


Common Queries
==============

Convenience functions for common queries.

If you are familiar with the neuPrint data model and the
`cypher <https://neo4j.com/developer/cypher-query-language/>`_
query language, you can write your own queries using
:py:func:`fetch_custom <fetch_custom>`.
But the functions in this file offer a convenient API for common queries.

.. autosummary::

    fetch_custom
    find_neurons
    fetch_custom_neurons
    simple_connections
    fetch_traced_adjacencies
    fetch_meta
    fetch_all_rois
    fetch_primary_rois

.. automodule:: neuprint.queries
   :members:

