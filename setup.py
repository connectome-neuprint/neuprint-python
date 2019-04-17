from setuptools import setup

import re


VERSIONFILE = "neuprint/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__verstr__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.startswith('#')]

setup(
    name='neuprint-python',
    version=verstr,
    description="Python client utilties for interacting with the neuPrint connectome analysis service",
    author="Philipp Schlegel",
    author_email='pms70@cam.ac.uk',
    url='https://github.com/schlegelp/neuprint-python',
    packages=['neuprint'],
    entry_points={},
    install_requires=requirements,
    keywords='neuprint-python',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
