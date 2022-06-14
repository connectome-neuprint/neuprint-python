#!/bin/bash

set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
REPO_DIR="$(dirname ${SCRIPT_DIR})"

echo "Building docs in your local repo"
cd ${REPO_DIR}/docs
GIT_DESC=$(git describe)
make html

TMP_REPO=$(mktemp -d)
echo "Cloning to ${TMP_REPO}/neuprint-python"
cd ${TMP_REPO}
git clone ssh://git@github.com/connectome-neuprint/neuprint-python 
cd neuprint-python

echo "Committing built docs"
git switch -c gh-pages origin/gh-pages
rm -r docs
cp -R ${REPO_DIR}/docs/build/html docs
git add .
git commit -m "Updated docs for ${GIT_DESC}" .

echo "Pushing to github"
git push origin gh-pages

echo "DONE deploying docs"
