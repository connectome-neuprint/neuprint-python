.. currentmodule:: neuprint.client

.. _client:


Client
======

.. automodule:: neuprint.client

.. autosummary::

    default_client
    set_default_client
    clear_default_client
    list_all_clients
    setup_debug_logging
    disable_debug_logging


:py:class:`Client` methods correspond directly to built-in
`neuprintHTTP API endpoints <https://neuprint.janelia.org/help/api?dataset=hemibrain%3Av1.2.1&qt=findneurons&q=1>`_.


.. autosummary::

    Client
    Client.fetch_custom
    Client.fetch_available
    Client.fetch_help
    Client.fetch_server_info
    Client.fetch_version
    Client.fetch_database
    Client.fetch_datasets
    Client.fetch_instances
    Client.fetch_db_version
    Client.fetch_profile
    Client.fetch_token
    Client.fetch_daily_type
    Client.fetch_roi_completeness
    Client.fetch_roi_connectivity
    Client.fetch_roi_mesh
    Client.fetch_skeleton
    Client.fetch_raw_keyvalue
    Client.post_raw_keyvalue

.. autoclass:: neuprint.client.Client
   :members:

.. autofunction:: default_client
.. autofunction:: set_default_client
.. autofunction:: clear_default_client
.. autofunction:: list_all_clients
.. autofunction:: setup_debug_logging
.. autofunction:: disable_debug_logging
