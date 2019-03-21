neuprint-python
===============================

Python client utilties for interacting with the neuPrint connectome analysis service.

This repository was forked from [connectome-neuprint/neuprint-python](https://github.com/connectome-neuprint/neuprint-python)
and simply adds some convenience functions to query data from a neuPrint
server.

## Install

Make sure you have [Python 3](https://www.python.org),
[pip](https://pip.pypa.io/en/stable/installing/) and
[git](https://git-scm.com) installed. Then run this in terminal:

```
pip install git+git://github.com/schlegelp/neuprint-python@master
```

## Getting started

First, grab your token from the neuPrint web interface

![token](examples/img/token-screenshot.png)

Next

```Python
import neuprint as neu

# Set up credentials
client = neu.Client('http://your.neuprintserver.com:8800', 'yourtoken')

# Grab some neurons by ROI
lh = neu.fetch_neurons_in_roi('LH')
```