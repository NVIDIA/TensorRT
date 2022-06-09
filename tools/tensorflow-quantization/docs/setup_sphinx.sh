#!/bin/bash

# Install Sphinx
python3 -m pip install Sphinx

# Install sphinx-rtd-theme and sphinx-glpi-theme
python3 -m pip install sphinx-rtd-theme
python3 -m pip install sphinx-glpi-theme

# Install myst-nb extension to process notebooks and myst markdown files
python3 -m pip install myst-nb

# Install mermaid for flow diagrams building
python3 -m pip install sphinxcontrib-mermaid
