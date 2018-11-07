from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='neuprint-python',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Python client utilties for interacting with the neuPrint connectome analysis service",
    author="Stuart Berg",
    author_email='bergs@janelia.hhmi.org',
    url='https://github.com/stuarteberg/neuprint-python',
    packages=['neuprint'],
    entry_points={
        'console_scripts': [
            'neuprint=neuprint.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='neuprint-python',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ]
)
