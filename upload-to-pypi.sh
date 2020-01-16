#!/bin/bash

##
## Usage: Run this from within the root of the repo.
##

if [[ "$(git describe)" == *-* ]]; then
    echo "Error:" 1>&2
    echo "  Can't package a non-tagged commit." 1>&2
    echo "  Your current git commit isn't tagged with a proper version." 1>&2
    echo "  Try 'git commit -a' first" 1>&2
    exit 1
fi

rm -rf dist build
python setup.py sdist bdist_wheel

# The test PyPI server
#python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# The real PyPI server
python3 -m twine upload dist/*
