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

"""
This file contains configurable interactive widget wrappers.
"""


from ipywidgets import widgets
import IPython
from IPython.display import display
from typing import List


class InteractiveDiagram:
    """A dropdown widget wrapper"""
    def __init__(self, diagram_renderer, choices, description):
        def get_default_choice_key():
            return list(self.choices.keys())[0]

        self.diagram_renderer = diagram_renderer
        self.choices = choices

        self.choice_widget = widgets.Dropdown(
            options=self.choices.keys(),
            value=get_default_choice_key(),
            description=description,
            disabled=False,
        )
        out = widgets.interactive_output(self.dropdown_state_eventhandler, {'user_choice': self.choice_widget})
        display(out)

    def _render(self, choice, values):
        display(self.choice_widget)
        self.diagram_renderer(choice, *values)

    def dropdown_state_eventhandler(self, user_choice):
        values = self.choices[user_choice]
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = (values,)
        self._render(user_choice, values)


class InteractiveDiagram_2:
    """A dropdown widget wrapper"""
    def __init__(self, choices: List, description: str):
        def get_default_choice():
            return list(self.choices.keys())[0]

        self.choices = choices
        self.choice_widget = widgets.Dropdown(
            options=self.choices.keys(),
            value=get_default_choice(),
            description=description,
            disabled=False,
        )
        out = widgets.interactive_output(self.dropdown_state_eventhandler, {'user_choice': self.choice_widget})
        display(out)

    def _render(self, title, renderer):
        display(self.choice_widget)
        renderer(title=title)

    def dropdown_state_eventhandler(self, user_choice):
        renderer = self.choices[user_choice]
        self._render(user_choice, renderer)