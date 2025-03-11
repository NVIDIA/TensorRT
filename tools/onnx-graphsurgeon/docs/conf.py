#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "myst_parser",
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
copyright = "2024 NVIDIA Corporation"
author = "NVIDIA"

version = gs.__version__
# The full version, including alpha/beta/rc tags.
release = version

# Style
pygments_style = "colorful"

html_theme = "furo"

html_title = f"{project}<br>{version}"

# Use the TRT theme and NVIDIA logo
html_static_path = ["_static"]

# Hide source link
html_show_sourcelink = False

# Output file base name for HTML help builder.
htmlhelp_basename = "OnnxGraphSurgeonDoc"

# Template files to extend default Sphinx templates.
# See https://www.sphinx-doc.org/en/master/templating.html for details.
templates_path = ["_templates"]

# For constructor arguments to show up in Sphinx generated doc
autoclass_content = "both"

html_theme_options = {
    "light_logo": os.path.join("img", "nvlogo_black.png"),
    "dark_logo": os.path.join("img", "nvlogo_white.png"),
    "light_css_variables": {
        "color-api-pre-name": "#4e9a06",
        "color-api-name": "#4e9a06",
        "color-api-background": "#e8e8e8",
    },
    "dark_css_variables": {
        "color-api-background": "#303030",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA/TensorRT",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}


# Allows us to override the default page width in the Sphinx theme.
def setup(app):
    app.add_css_file("style.css")
