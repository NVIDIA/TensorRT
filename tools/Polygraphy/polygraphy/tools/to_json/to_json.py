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
from polygraphy import mod
from polygraphy.json import save_json
from polygraphy.tools.base import Tool


class ToJSON(Tool):
    """
    [TEMPORARY] Converts pickled data to JSON.
    This tool will be removed in 0.31.0 since all future versions of Polygraphy
    will not use Pickle for serialization.
    """

    def __init__(self):
        mod.warn_deprecated("to-json", use_instead="JSON serialization", remove_in="0.31.0")
        super().__init__(name="to-json")

    def add_parser_args(self, parser):
        parser.add_argument("pickle_data", help="Path to old pickled data")
        parser.add_argument("-o", "--output", help="Path at which to write the JSON-ified data.", required=True)

    def run(self, args):
        import pickle

        import polygraphy
        from polygraphy.comparator.struct import RunResults

        class LegacyRunResults(list):
            pass

        polygraphy.comparator.struct.RunResults = LegacyRunResults

        with open(args.pickle_data, "rb") as f:
            data = pickle.load(f)

            if isinstance(data, LegacyRunResults):
                data = RunResults(list(data))

            save_json(data, args.output)
