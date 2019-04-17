neuprint-python
===============================
Python client utilties for interacting with the neuPrint connectome analysis service

See below to get started, or have a look at the [Introduction notebook](examples/Introduction.ipynb).

Find the documentation [here](https://neuprint-python.readthedocs.io)!

## Install

If you're using conda, use this command:

```shell
conda install -c flyem-forge neuprint-python
```

Otherwise, make sure you have [Python 3](https://www.python.org),
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

