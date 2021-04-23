#!/bin/bash
export FLASK_APP=nlp260app.py
export FLASK_ENV=development

if ! [ -e "$FLASK_APP" ]; then
    echo "ERROR: '$FLASK_APP' does not exist"
    exit 1
fi

python -m flask run
