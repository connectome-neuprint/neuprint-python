neuprint-python
===============

.. _intro:


Introduction to neuPrint and ``neuprint-python``
------------------------------------------------

The `neuPrint project <https://github.com/connectome-neuprint>`_ defines
a graph database structure and suite of tools for storing and analyzing
connectomic data.

The best way to become acquainted with neuPrint's capabilities and data
model is to experiment with a public neuprint database via the neuprint
web UI.  Try exploring the `Janelia FlyEM Hemibrain neuprint database <https://neuprint.janelia.org/>`_.


Once you're familiar with the basics, you're ready to start writing
Python scripts to query the database programmatically with 
``neuprint-python``.

.. _install:

Install neuprint-python
-----------------------

If you're using `conda <https://docs.conda.io/en/latest/>`_, use this command:


.. code-block:: bash

    conda install -c flyem-forge neuprint-python


Otherwise, use ``pip``:


.. code-block:: bash

    pip install neuprint-python

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   client
   queries
   admin
   deprecated
