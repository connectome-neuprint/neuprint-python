neuprint-python
===============

``neuprint-python`` allows you to programmatically interact with the
`neuPrint <https://github.com/connectome-neuprint>`_ connectome analysis
service.

Install
-------

Make sure you have `Python 3 <https://www.python.org>`_,
`pip <https://pip.pypa.io/en/stable/installing/>`_ and
`git <https://git-scm.com>`_ installed. Then run this in terminal:

::

    pip install git+git://github.com/connectome-neuprint/neuprint-python@master


What can ``neuprint-python`` do for you?
----------------------------------------
This library mirrors some but not all of the widgets in the neuprint
weblclient. Currently you can for example:

- :func:`find <neuprint.fetch.find_neurons>` neurons based on name/body IDs
- get neurons within a given :func:`roi <neuprint.fetch.fetch_neurons_in_roi>`
- fetch custom :func:`cyphers <neuprint.fetch.fetch_custom>`
- fetch :func:`connectivity <neuprint.fetch.fetch_connectivity>` table

Check out the full :doc:`API </src/api>` and the examples below.

Quickstart
----------
First, you will need your API token. You can get it through the neuPrint
website.

.. image:: ../examples/img/token-screenshot.png
   :width: 50%
   :alt: token screenshot
   :align: left


Once you have your token, fire up Python and import neuprint.
::

    import neuprint as neu

    client = neu.Client('https://your.neuprintserver.com:8800', 'yourtoken')


**Important**: your client's authentication can go stale which will result
in a ``ConnectionError``. In that case simply reinitalise the client
(``client -  neu.Client(....``).

You should be all set now, so try this::

    # Find all neurons with "MBON" in their name
    mbons = neu.find_neurons('.*MBON.*')
    mbons.head()


Examples
--------

Above quickstart example illustrated the use of
:func:`~neuprint.fetch.find_neurons`. The result will be a DataFrame
containing body Ids, status, names, etc. Most neuprint functions allow you
to ask for additional properties::


    # Find MBONs and retrieve their soma location
    mbons = neu.find_neurons('.*MBON.*', add_props=['somaLocation'])
    mbons.head()


More examples to come soon!
