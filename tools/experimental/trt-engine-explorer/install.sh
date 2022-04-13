#!/bin/sh

python3 -m virtualenv env_trex
source env_trex/bin/activate
python3 -m pip install -e .
jupyter nbextension enable widgetsnbextension --user --py
