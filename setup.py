from setuptools import setup

import versioneer

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.strip().startswith('#')]

setup(
    name='neuprint-python',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Python client utilties for interacting with the neuPrint connectome analysis service",
    author="Philipp Schlegel",
    author_email='pms70@cam.ac.uk',
    url='https://github.com/connectome-neuprint/neuprint-python',
    packages=['neuprint'],
    entry_points={},
    install_requires=requirements,
    keywords='neuprint-python',
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ]
)
