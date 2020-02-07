import os

NEUPRINT_SERVER = 'neuprint.janelia.org'
DATASET = 'hemibrain:v1.0.1'

try:
    TOKEN = os.environ['NEUPRINT_APPLICATION_CREDENTIALS']
except KeyError:
    raise RuntimeError("These tests assume that NEUPRINT_APPLICATION_CREDENTIALS is defined in your environment!")

