.. currentmodule:: neuprint.client

.. _quickstart:

Quickstart
==========

Install neuprint-python
-----------------------

If you're using `conda <https://docs.conda.io/en/latest/>`_, use this command:


.. code-block:: bash

    conda install -c flyem-forge neuprint-python


Otherwise, use ``pip``:


.. code-block:: bash

    pip install neuprint-python

Client and Authorization Token
------------------------------

All communication with the ``neuPrintHTTP`` server is done via a :py:class:`Client` object.

To create a :py:class:`Client`, you must provide three things:

    - The neuprint server address (e.g. ``neuprint.janelia.org``)
    - Which dataset you'll be fetching from (e.g. ``hemibrain:v1.2.1``)
    - Your personal authentication token

To obtain your authorization token, follow these steps:

    1. Navigate your web browser to the neuprint server address.
    2. Log in.
    3. Using the account menu in the upper right-hand corner, select "Account" as shown in the screenshot below.
    4. Copy the entire auth token.


.. image:: _static/token-screenshot.png
   :scale: 50  %
   :alt: Auth Token menu screenshot

Create the Client
-----------------

    .. code-block:: python

        from neuprint import Client

        c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token='YOUR-TOKEN-HERE')
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
           ...:     RETURN n.bodyId AS bodyId, n.type as type, n.instance AS instance, n.pre AS numpre, n.post AS numpost
           ...:     ORDER BY n.pre + n.post DESC
           ...: """

        In [2]: results = c.fetch_custom(q)

        In [3]: print(f"Found {len(results)} results")
        Found 177 results

        In [4]: results.head()
        Out[4]:
               bodyId    type                   instance  numpre  numpost
        0  5813027016    FB4Y             FB4Y(EB/NO1)_R    1720     6508
        1  1008378448    FB4Y             FB4Y(EB/NO1)_R    1791     6301
        2  1513363614  LCNOpm        LCNOpm(LAL-NO3pm)_R     858     6501
        3  5813057274    FB4Y             FB4Y(EB/NO1)_L    2001     5089
        4  1131827390    FB4M  FB4M(PPM3-FB3/4-NO-DAN)_R    2614     4431

Next Steps
----------

Try the `interactive tutorial`_ for a tour of basic features in ``neuprint-python``.

.. _interactive tutorial: notebooks/QueryTutorial.ipynb
