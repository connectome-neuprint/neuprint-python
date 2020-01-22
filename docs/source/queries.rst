.. currentmodule:: neuprint

.. _queries:


Queries
=======

Convenience functions for common queries.

If you are familiar with the neuPrint data model and the
`cypher <https://neo4j.com/developer/cypher-query-language/>`_
query language, you can write your own queries using
:py:func:`fetch_custom <fetch_custom>`.
But the functions in this file offer a convenient API for common queries.

..
   (The following |br| definition is the only way
   I can force numpydoc to display explicit newlines...) 

.. |br| raw:: html

   <br />


.. autofunction:: fetch_custom
.. autofunction:: custom_search

.. autofunction:: fetch_neurons_in_roi
.. autofunction:: find_neurons
.. autofunction:: fetch_connectivity
.. autofunction:: fetch_connectivity_in_roi
.. autofunction:: fetch_edges
.. autofunction:: fetch_synapses

