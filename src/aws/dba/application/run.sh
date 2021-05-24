#!/bin/bash
export FLASK_APP=DBAapp.py

# docker container is launched from /, so we need to cd to the relevant code.
[[ "$(whoami)" == "root" ]] && cd /application

if ! [ -e "$FLASK_APP" ]; then
    echo "ERROR: '$FLASK_APP' does not exist"
    exit 1
fi


if [[ "$(whoami)" != "root" ]]; then
    export FLASK_ENV=development

    # attempt to give application access to 'lib'
    export PYTHONPATH="$PYTHONPATH:../dba_scripts"

    python -m flask run
else
    export FLASK_ENV=development

    # give application access to 'lib'
    export PYTHONPATH="$PYTHONPATH:/dba_scripts"

    python -m flask run -h 0.0.0.0 -p 81
fi
