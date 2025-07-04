.. _development:

Development Notes
=================

Notes for maintaining ``neuprint-python``.

Prerequisites
-------------

**For general development:**

Make sure you have both ``flyem-forge`` and ``conda-forge`` listed as channels in your ``.condarc`` file.
(If you don't know where your ``.condarc`` file is, check ``conda config --show-sources``.)

.. code-block:: yaml

    # .condarc
    channels:
    - flyem-forge
    - conda-forge
    - nodefaults  # A magic channel that forbids any downloads from the anaconda default channels.

**For packaging and release:**

Install ``conda-build`` if you don't have it yet:

.. code-block:: bash

    conda install -n base conda-build anaconda-client twine setuptools


Before you can upload packages to anaconda.org, you'll need to be a member of the ``flyem-forge`` organization.
Then you'll need to run ``anaconda login``.

Before you can upload packages to PyPI, you'll need to be added as a "collaborator" of the
``neuprint-python`` project on PyPI.  Then you'll need to log in and obtain a token with
an appropriate scope for ``neuprint-python`` and add it to your ``~/.pypirc`` file:

.. code-block::

    [distutils]
    index-servers =
        neuprint-python
        my-other-project

    [neuprint-python]
    repository = https://upload.pypi.org/legacy/
    username = __token__
    password = <your token goes here>

    [my-other-project]
    repository = https://upload.pypi.org/legacy/
    username = __token__
    password = <your other token goes here>


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
    git commit -m "Updated changelog" docs/source/changelog.rst

    # Do the tests still pass?
    pytest .

    # Do the docs still build?
    (
        export PYTHONPATH=$(pwd)
        cd docs
        make html
        open build/html/index.html
    )

    # Tag the git repo with the new version
    NEW_TAG=0.3.1
    git tag -a ${NEW_TAG} -m ${NEW_TAG}
    git push --tags origin

    # Build and upload the conda package
    conda build conda-recipe
    anaconda upload -u flyem-forge $(conda info --base)/conda-bld/noarch/neuprint-python-${NEW_TAG}-py_0.tar.bz2

    # Build and upload the PyPI package
    ./upload-to-pypi.sh

    # Deploy the docs
    ./docs/deploy-docs.sh


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

The docs are built with Sphinx.  See ``docs/requirements.txt`` for the docs dependencies (in addition to those listed in ``dependencies.txt``). In addition, building the docs requires the ``pandoc`` command-line tool to be installed in the environment (the Python wrapper, though, is not required).

The example notebooks are run when the docs are built. For this to succeed, the ``neuprint`` library must be on your ``PYTHONPATH``. Like the tests, the docs rely on the public ``hemibrain:v1.2.1`` dataset on ``neuprint.janelia.org``, which means you must define ``NEUPRINT_APPLICATION_CREDENTIALS`` in your environment before running them.

To build the docs locally:

.. code-block:: bash

    cd neuprint-python/docs
    make html
    open build/html/index.html

We publish the docs via `github pages <https://pages.github.com/>`_.
Use the script ``docs/deploy-docs.sh`` to build and publish the docs to GitHub in the `gh-pages` branch.
(At some point in the future, we may automate this via a CI system.)

.. code-block:: bash

    ./docs/deploy-docs.sh


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
