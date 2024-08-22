from setuptools import setup, find_packages

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
    packages=find_packages(),
    entry_points={},
    install_requires=requirements,
    keywords='neuprint-python',
    python_requires='>=3.9',
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ]
)
