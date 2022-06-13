.. _development:

Development Notes
=================

Notes for maintaining ``neuprint-python``.

Packaging and Release
---------------------

``neuprint-python`` is packaged for both ``conda`` (on the `flyem-forge channel <https://anaconda.org/flyem-forge/neuprint-python/files>`_)
and ``pip`` (on `PyPI <https://pypi.org/project/neuprint-python/>`_).

The package version is automatically inferred from the git tag.
To prepare a release, follow these steps:

.. code-block:: bash

    cd neuprint-python

    # Update the change log!
    code docs/source/changelog.rst

    # Do the tests still pass?
    pytest .

    # Do the docs still build?
    cd docs && make html && cd -

    # Tag the git repo with the new version
    NEW_TAG=0.3.1
    git tag -a ${NEW_TAG} -m ${NEW_TAG}
    git push --tags origin

    # Build and upload the conda package
    conda build conda-recipe
    anaconda upload -u flyem-forge $(conda info --base)/conda-bld/noarch/neuprint-python-${NEW_TAG}-py_0.tar.bz2

    # Build and upload the PyPI package
    ./upload-to-pypi.sh

    # Update the docs branch (see explanation below)
    git branch -D docs
    git branch docs ${NEW_TAG}
    git push -f origin docs


Dependencies
------------

If you need to add dependencies to ``neuprint-python``, edit ``dependencies.txt`` (which is used by the conda recipe).
You should also update ``environment.yml`` so that our binder container will acquire the new dependencies
when users try out the interactive `tutorial`_.  After publishing a new conda package with the updated dependencies,
follow these steps **on a Linux machine**:

.. code-block:: bash

    #!/bin/bash
    # update-deps.sh

    set -e

    # Create an environment with the binder dependencies
    TUTORIAL_DEPS="ipywidgets bokeh holoviews hvplot"
    SIMULATION_DEPS="ngspice umap-learn scikit-learn matplotlib"
    BINDER_DEPS="neuprint-python jupyterlab ${TUTORIAL_DEPS} ${SIMULATION_DEPS}"
    conda create -y -n neuprint-python -c flyem-forge -c conda-forge ${BINDER_DEPS}

    # Export to environment.yml, but relax the neuprint-python version requirement
    conda env export -n neuprint-python > environment.yml
    sed --in-place 's/neuprint-python=.*/neuprint-python/g' environment.yml

    git commit -m "Updated environment.yml for binder" environment.yml
    git push origin master


.. _tutorial: notebooks/QueryTutorial.ipynb

Documentation
-------------

The docs are built with Sphinx.  See ``docs/requirements.txt`` for the docs dependencies.
To build the docs locally:

.. code-block:: bash

    cd neuprint-python/docs
    make html
    open build/html/index.html

The Travis-CI build will automatically deploy the docs to github pages using `doctr <https://github.com/drdoctr/doctr/>`_,
every time you push to the the ``docs`` branch.  To sync that branch with a recent tag, try:

.. code-block:: bash

    cd neuprint-python
    LATEST_TAG=$(git tag -l --sort=v:refname | tail -n1)
    git branch -D docs
    git branch docs ${LATEST_TAG}
    git push -f origin docs

The docs are also built for development branches, but they're deployed to a special location:
``https://connectome-neuprint.github.io/neuprint-python/docs-<BRANCH_NAME>``

.. warning::

    The docs are rebuilt and published every time you push to the master branch.
    If you push (and document) API-breaking changes without publishing new packages,
    the documentation will not correspond to the ``conda`` and ``pip`` packages!
    Only push to master when you're ready to deploy new packages.


Interactive Tutorial
--------------------

The documentation contains a `tutorial`_ which can be launched interactively via binder.
To update the tutorial contents, simply edit the ``.ipynb`` file and re-build the docs.

If the binder setup is broken, make sure the dependencies are configured properly as described above.

It takes a few minutes to initialize the binder container for the first time after a new release.
Consider sparing your users from that by clicking the binder button yourself after each release.

Tests
-----

The tests require ``pytest``, and they rely on the public ``hemibrain:v1.2.1`` dataset on ``neuprint.janelia.org``,
which means you must define ``NEUPRINT_APPLICATION_CREDENTIALS`` in your environment before running them.

To run the tests:

.. code-block:: bash

    cd neuprint-python
    PYTHONPATH=. pytest neuprint/tests
