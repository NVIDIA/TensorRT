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

"""
This file contains configurable interactive widget wrappers.
"""


from ipywidgets import widgets
import IPython
from IPython.core.display import display
from typing import List


class InteractiveDiagram:
    """A dropdown widget wrapper"""
    def __init__(self, diagram_renderer, choices, description):
        def get_default_choice_key():
            return list(self.choices.keys())[0]

        def get_default_choice_value():
            val = list(self.choices.values())[0]
            if not isinstance(val, list) and not isinstance(val, tuple):
                return (val,)
            return val

        self.diagram_renderer = diagram_renderer
        self.choices = choices

        display(get_default_choice_key())

        self.choice_widget = widgets.Dropdown(
            options=self.choices.keys(),
            value=get_default_choice_key(),
            description=description,
            disabled=False,
        )
        dropdown_state_eventhandler = lambda change: self.dropdown_state_eventhandler(change)
        self.choice_widget.observe(dropdown_state_eventhandler, names='value')
        self._render(get_default_choice_key(), get_default_choice_value())

    def _render(self, choice, values):
        IPython.display.clear_output(wait=True)
        display(self.choice_widget)
        self.diagram_renderer(choice, *values)

    def dropdown_state_eventhandler(self, change):
        state_choice = change.new
        values = self.choices[state_choice]
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = (values,)
        self._render(state_choice, values)


class InteractiveDiagram_2:
    """A dropdown widget wrapper"""
    def __init__(self, choices: List, description: str):
        def get_default_choice():
            return list(self.choices.keys())[0]

        def get_default_renderer():
            val = list(self.choices.values())[0]
            return val

        self.choices = choices
        display(get_default_choice())

        self.choice_widget = widgets.Dropdown(
            options=self.choices.keys(),
            value=get_default_choice(),
            description=description,
            disabled=False,
        )
        dropdown_state_eventhandler = lambda change: self.dropdown_state_eventhandler(change)
        self.choice_widget.observe(dropdown_state_eventhandler, names='value')
        self._render(get_default_choice(), get_default_renderer())

    def _render(self, title, renderer):
        IPython.display.clear_output(wait=True)
        display(self.choice_widget)
        renderer(title=title)

    def dropdown_state_eventhandler(self, change):
        state_choice = change.new
        renderer = self.choices[state_choice]
        self._render(state_choice, renderer)
