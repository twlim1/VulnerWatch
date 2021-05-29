#!/bin/bash
export FLASK_APP=nlp260app.py

# docker container is launched from /, so we need to cd to the relevant code.
[[ "$(whoami)" == "root" ]] && cd /application

if ! [ -e "$FLASK_APP" ]; then
    echo "ERROR: '$FLASK_APP' does not exist"
    exit 1
fi

if [[ "$(whoami)" != "root" ]]; then
    export FLASK_ENV=development
    python -m flask run
else
    export FLASK_ENV=development
    python -m flask run -h 0.0.0.0 -p 80
fi
