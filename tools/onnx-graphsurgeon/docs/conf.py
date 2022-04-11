#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import sys
import os

ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.path.pardir)
sys.path.insert(0, ROOT_DIR)
import onnx_graphsurgeon as gs

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

# Want to be able to generate docs with no dependencies installed
autodoc_mock_imports = ["onnx", "numpy", "onnxruntime"]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "special-members": "__call__, __getitem__, __bool__",
}

autodoc_member_order = "bysource"

autodoc_inherit_docstrings = True

autosummary_generate = True

source_suffix = [".rst"]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "ONNX GraphSurgeon"
copyright = "2020, NVIDIA"
author = "NVIDIA"

version = gs.__version__
# The full version, including alpha/beta/rc tags.
release = version

# Style
pygments_style = "colorful"

html_theme = "sphinx_rtd_theme"

# Use the TRT theme and NVIDIA logo
html_static_path = ["_static"]

html_logo = "_static/img/nvlogo_white.png"

# Hide source link
html_show_sourcelink = False

# Output file base name for HTML help builder.
htmlhelp_basename = "OnnxGraphSurgeonDoc"

# Template files to extend default Sphinx templates.
# See https://www.sphinx-doc.org/en/master/templating.html for details.
templates_path = ["_templates"]

# For constructor arguments to show up in Sphinx generated doc
autoclass_content = "both"

# Unlimited depth sidebar.
html_theme_options = {"navigation_depth": -1}

html_sidebars = {"**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]}

# Allows us to override the default page width in the Sphinx theme.
def setup(app):
    app.add_css_file("style.css")
