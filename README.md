[![docs-badge](docs/source/_static/docs-badge.svg)][docs]

neuprint-python
===============

Python client utilties for interacting with the [neuPrint][neuprint] connectome analysis service.

[neuprint]: https://neuprint.janelia.org

## Install

If you're using pixi, use this:
```shell
pixi init -c flyem-forge -c conda-forge
pixi add python=3.9 'neuprint-python>=0.5.1' 'pyarrow>=20' 'numpy>=2' 'pandas>=2'
```

If you're using conda, use this command:

```shell
conda install -c flyem-forge neuprint-python
```

Otherwise, use pip:

```shell
pip install neuprint-python
```

## Getting started

See the [Quickstart section][quickstart] in the [documentation][docs]

[docs]: http://connectome-neuprint.github.io/neuprint-python/docs/
[quickstart]: http://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html

