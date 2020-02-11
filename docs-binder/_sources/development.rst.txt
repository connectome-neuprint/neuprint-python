.. _development:

Development Notes
=================

Notes for maintaining ``neuprint-python``.

Packaging
---------

``neuprint-python`` is packaged for both ``conda`` (on the `flyem-forge channel <https://anaconda.org/flyem-forge/neuprint-python/files>`_)
and ``pip`` (on `PyPI <https://pypi.org/project/neuprint-python/>`_.

The package version is automatically inferred from the git tag.
To prepare a release, follow these steps:

    .. code-block:: bash
    
        cd neuprint-python
    
        # Tag the git repo with the new version
        git tag -a 0.3.1 -m 0.3.1
        git push --tags origin
        
        # Build and upload the conda package
        conda build conda-recipe
        anaconda upload -u flyem-forge $(conda info --base)/conda-bld/noarch/neuprint-python-0.3.1-py_0.tar.bz2

        # Build and upload the PyPI package
        ./upload-to-pypi.sh        

Documentation
-------------

The docs are built with Sphinx.  See ``docs/requirements.txt`` for the docs dependencies.
To build the docs locally:

    .. code-block:: bash
    
        cd neuprint-python/docs
        make html
        open build/html/index.html

The Travis-CI build will automatically deploy the docs to github pages using `doctr <https://github.com/drdoctr/doctr/>`_,
every time you push to the the master branch.
The docs are also built for development branches, but they're deployed to a special location:
``https://connectome-neuprint.github.io/neuprint-python/docs-<BRANCH_NAME>``

    .. warning::
    
        The docs are rebuilt and published every time you push to the master branch.
        If you push (and document) API-breaking changes without publishing new packages,
        the documentation will not correspond to the ``conda`` and ``pip`` packages!
        Only push to master when you're ready to deploy new packages.

Tests
-----

The tests require ``pytest``, and they rely on the public ``hemibrain:v1.0`` dataset on ``neuprint.janelia.org``,
which means you must define ``NEUPRINT_APPLICATION_CREDENTIALS`` in your environment before running them.

To run the tests:

    .. code-block:: bash
    
        cd neuprint-python
        PYTHONPATH=. pytest neuprint/tests
