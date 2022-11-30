#!/bin/bash

set -e

# the dir containing this script is the base
export BASEDIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ;  pwd -P)"

export NOTEBOOK_PATH="${BASEDIR}/notebooks"
SOME_RANDOM_PORT="18$(( ( RANDOM % 10 ) ))$(( ( RANDOM % 10 ) ))$(( ( RANDOM % 10 ) ))"
# EXPOSED_PORT is the single open port when running in a docker container
export NOTEBOOK_PORT=${EXPOSED_PORT:-$SOME_RANDOM_PORT}


# --allow-root is required for the root user in a rootless docker container
# --ip=0.0.0.0 is required to allow access from outside a docker container
jupyter-notebook --config="${BASEDIR}/.jupyter_notebook_config.py" --allow-root --ip=0.0.0.0 --no-browser --notebook-dir="${NOTEBOOK_PATH}" --port=$NOTEBOOK_PORT
