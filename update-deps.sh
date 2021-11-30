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
