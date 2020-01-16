neuprint-python
===============================
Python client utilties for interacting with the neuPrint connectome analysis service

See below to get started, or have a look at the [Introduction notebook][intro].

[intro]: https://github.com/connectome-neuprint/neuprint-python/tree/master/examples/Introduction.ipynb

Find the documentation [here](https://neuprint-python.readthedocs.io)!

## Install

If you're using conda, use this command:

```shell
conda install -c flyem-forge neuprint-python
```

Otherwise, use pip:

```shell
pip install neuprint-python
```

## Getting started

First, grab your token from the neuPrint web interface.

![token](https://raw.githubusercontent.com/connectome-neuprint/neuprint-python/master/examples/img/token-screenshot.png)

Next:

```Python
import neuprint

# Set up credentials
client = neuprint.Client('https://neuprint-test.janelia.org', 'your-token-here')

# Grab some neurons by ROI.
# In this case, get all neurons intersecting the Lateral Horn (right side).
q = """
    MATCH (neuron :hemibrain_Neuron)
    WHERE (neuron.`LH(R)`)
    RETURN neuron.bodyId, neuron.instance, neuron.status, neuron.cropped
"""

lh_table = neuprint.fetch_custom(q)
print(lh_table.head())
```

```
   neuron.bodyId        neuron.instance neuron.status neuron.cropped
0      420594200  put_ADL11c_a(ADL11)_R        Traced          False
1      792692885              SCL-SLP_R        Traced          False
2      850233586  PDL17c_b_pct(PDL17)_R        Traced          False
3      359279388    put_PDL10h(PDL10)_R        Traced          False
4      296120593   put_ADL10oa(ADL10)_R        Traced          False
```
