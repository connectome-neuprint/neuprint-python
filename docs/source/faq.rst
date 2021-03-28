.. _faq:

FAQ
===

Why use this API? Why not just use plain Cypher?
------------------------------------------------

Cypher is a powerful language for querying the neuprint database,
and there will always be some needs that can only be satisfed with
a custom-tailored Cypher query.

However, there are some advantages that come from using the higher-level
API provided in ``neuprint-python``:

* To use Cypher, you need an understanding of the neuprint data model.
  It's not too complex, but for many users, basic neuron attributes and
  connection information is enough.
* Some queries are difficult to specify.  For example, efficiently filtering neurons
  by ``inputRoi`` or ``outputRoi`` is not trivial.  But ``NeuronCriteria`` handles that for you.
* The ``neuprint-python`` API uses reasonable default parameters,
  which aren't always obvious in raw Cypher queries.
* ``neuprint-python`` saves you from certain nuisance tasks, like converting ``roiInfo``
  from JSON data into a DataFrame for easy analysis.
* When a query might return a large amount of data, it's often critical to break the query
  into batches, to avoid timeouts from the server.  For functions in which that is likely to occur,
  ``neuprint-python`` implements batching for you.

Nonetheless, if you need to run a query that isn't conveniently
supported by the high-level API in this library,
or you simply prefer to write your own Cypher,
then feel free to use :py:meth:`.Client.fetch_custom()`.


What Cypher queries are being used by this code internally?
-----------------------------------------------------------

Enable debug logging to see the cypher queries that are being sent to the neuPrint server.
See :py:func:`.setup_debug_logging()` for details.


Where are the release notes for the *data*?
-------------------------------------------

Please see the `neuprint dataset release notes and errata <https://neuprint.janelia.org/releasenotes>`_.


How can I download the exact Hemibrain ROI shapes?
--------------------------------------------------

A volume containing the exact primary ROI region labels for the hemibrain in hdf5 format can be `found here`_.
Please see the enclosed README for details on how to read and interpret the volume.

.. note::

   The volume tarball is only 10MB to download, but loading the full uncompressed volume requires 2 GB of RAM.

.. _found here: https://storage.cloud.google.com/hemibrain/v1.1/hemibrain-v1.1-primary-roi-segmentation.tar.gz


Can this library be used with ``multiprocessing``?
--------------------------------------------------

Yes. ``neuprint-python``'s mechanism for selecting the "default" client will automatically
copy the default client once per thread/process if necessary.  Thus, as long you're not
explicitly passing a ``client`` to any ``neuprint`` queries, your code can be run in
a ``threading`` or ``multiprocessing`` context without special care.
But if you are *not* using the default client, then it's your responsibility to create
a separate client for each thread/process in your program.
(``Client`` objects cannot be shared across threads or processes.)

.. note::

    Running many queries in parallel can place a heavy load the neuprint server.
    Please be considerate to other users, and limit the number of parallel queries you make.


Where can I find help?
----------------------

    - Please report issues and feature requests for ``neuprint-python`` on
      `github <https://github.com/connectome-neuprint/neuprint-python/issues>`_.

    - General questions about neuPrint or the hemibrain dataset can be asked on the neuPrint
      `Google Groups forum <https://groups.google.com/forum/#!forum/neuprint>`_.

    - For information about the Cypher query language, see the
      `neo4j docs <https://neo4j.com/developer/cypher-query-language/>`_.

    - The best way to become acquainted with neuPrint's capabilities and data
      model is to experiment with a public neuprint database via the neuprint
      web UI.  Try exploring the `Janelia FlyEM Hemibrain neuprint database <https://neuprint.janelia.org/>`_.
      To see the Cypher query that was used for each result on the site,
      click the information icon (shown below).

      .. image:: _static/neuprint-explorer-cypher-button.png
         :scale: 25  %
         :alt: Neuprint Explorer Cypher Info Button

