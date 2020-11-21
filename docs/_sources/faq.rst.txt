.. _faq:

FAQ
===

What Cypher Queries is my code using?
-------------------------------------

Enable debug logging to see the cypher queries that are being sent to the neuPrint server.
See :py:func:`.setup_debug_logging()` for details.

How can I download the exact ROI shapes?
----------------------------------------

A volume containing the exact primary ROI region labels in hdf5 format can be `found here`_.
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

