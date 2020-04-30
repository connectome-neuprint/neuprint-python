.. currentmodule:: neuprint.client

.. _quickstart:

Quickstart
==========


Client and Authorization Token
------------------------------

All communication with the ``neuPrintHTTP`` server is done via a :py:class:`Client` object.

To create a :py:class:`Client`, you must provide three things:

    - The neuprint server address (e.g. ``neuprint.janelia.org``)
    - Which dataset you'll be fetching from (e.g. ``hemibrain:v1.0``)
    - Your personal authentication token

To obtain your authorization token, follow these steps:

    1. Navigate your web browser to the neuprint server address.
    2. Log in.
    3. Using the account menu in the upper right-hand corner, select "Auth Token" as shown in the screenshot below.
    4. Copy the entire token


.. image:: _static/token-screenshot.png
   :scale: 50  %
   :alt: Auth Token menu screenshot

Create the Client
-----------------
   
    .. code-block:: python
   
        from neuprint import Client
    
        c = Client('neuprint.janelia.org', dataset='hemibrain', token='YOUR-TOKEN-HERE')
        c.fetch_version()

Alternatively, you can set your token in the following environment variable, in which case the ``token`` parameter can be omitted:


    .. code-block:: shell

        $ export NEUPRINT_APPLICATION_CREDENTIALS=<my-token>


Execute a query
---------------

Use your :py:class:`Client` to request data from neuprint.

The :py:meth:`Client.fetch_custom()` method will run an arbitrary cypher query against the database.
For information about the neuprint data model, see the `neuprint explorer web help. <https://neuprint.janelia.org/help>`_

Also, ``neuprint-python`` comes with convenience functions to implement common queries. See :ref:`queries`.

    .. code-block:: ipython


        In [1]: ## This query will return all neurons in the ROI ‘AB’
           ...: ## that have greater than 10 pre-synaptic sites.
           ...: ## Results are ordered by total synaptic sites (pre+post).
           ...: q = """\
           ...:     MATCH (n :Neuron {`AB(R)`: true})
           ...:     WHERE n.pre > 10
           ...:     RETURN n.bodyId AS bodyId, n.name AS name, n.pre AS numpre, n.post AS numpost
           ...:     ORDER BY n.pre + n.post DESC
           ...: """
        
        In [2]: results = c.fetch_custom(q)

        In [3]: print(f"Found {len(results)} results")
        Found 177 results

        In [4]: results.head()
        Out[4]:
               bodyId  name  numpre  numpost
        0  5813027016  None    1720     6484
        1  1008378448  None    1791     6276
        2  1513363614  None     858     6508
        3  5813057274  None    2001     5094
        4  1131827390  None    2614     4421

Next Steps
----------

Try the `interactive tutorial`_ for a tour of basic features in ``neuprint-python``.

.. _interactive tutorial: notebooks/QueryTutorial.ipynb
