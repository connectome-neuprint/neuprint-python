[![Documentation Status](https://readthedocs.org/projects/neuprint-python/badge/?version=latest)](http://neuprint-python.readthedocs.io/en/latest/?badge=latest)

neuprint-python
===============

Python client utilties for interacting with the
[neuPrint](https://github.com/connectome-neuprint) connectome analysis service.

This repository was forked from [connectome-neuprint/neuprint-python](https://github.com/connectome-neuprint/neuprint-python)
and simply adds some convenience functions to query data from a neuPrint
server.

If you want to query a DVID server, check out [dvidtools](https://github.com/flyconnectome/dvid_tools)
instead.

Find the documentation [here](https://neuprint-python.readthedocs.io)!

## Install

Make sure you have [Python 3](https://www.python.org),
[pip](https://pip.pypa.io/en/stable/installing/) and
[git](https://git-scm.com) installed. Then run this in terminal:

```shell
pip install git+git://github.com/schlegelp/neuprint-python@master
```

## Getting started

First, grab your token from the neuPrint web interface.

![token](examples/img/token-screenshot.png)

Next:

```Python
import neuprint as neu

# Set up credentials
client = neu.Client('https://your.neuprintserver.com:8800', 'yourtoken')

# Grab some neurons by ROI
lh = neu.fetch_neurons_in_roi('LH')
```