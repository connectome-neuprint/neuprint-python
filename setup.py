from setuptools import setup

import versioneer

with open('dependencies.txt') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.strip().startswith('#')]

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='neuprint-python',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Python client utilties for interacting with the neuPrint connectome analysis service",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Stuart Berg",
    author_email='bergs@hhmi.janelia.org',
    url='https://github.com/connectome-neuprint/neuprint-python',
    packages=['neuprint', 'neuprint.deprecated', 'neuprint.tests'],
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
