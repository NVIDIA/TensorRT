#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Tests that ensure the user experience is nice. For example, making sure that
README links work.
"""

import glob
import os
import re

import pytest
import requests

from tests.helper import ROOT_DIR

readme_test_cases = glob.glob(os.path.join(ROOT_DIR, "**", "*.md"), recursive=True)


class TestReadme(object):
    @pytest.mark.parametrize("readme", readme_test_cases)
    def test_links_valid(self, readme):
        MD_LINK_PAT = re.compile(r"\[.*?\]\((.*?)\)")

        readme_dir = os.path.dirname(readme)
        with open(readme, "r") as f:
            links = MD_LINK_PAT.findall(f.read())

        for link in links:
            link, _, _ = link.partition("#")  # Ignore section links for now
            if link.startswith("https://"):
                assert requests.get(link).status_code == 200
            else:
                assert os.path.pathsep * 2 not in link, "Duplicate slashes break links in GitHub"
                link_abs_path = os.path.abspath(os.path.join(readme_dir, link))
                assert os.path.exists(
                    link_abs_path
                ), "In README: '{:}', link: '{:}' does not exist. Note: Full path was: '{:}'".format(
                    readme, link, link_abs_path
                )
