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

from polygraphy.logger import G_LOGGER
from polygraphy.tools.base import Tool
from polygraphy.tools.debug.subtool.artifact_sorter import ArtifactSorterArgs


class Repeat(Tool):
    """
    [EXPERIMENTAL] Run an arbitrary command repeatedly, sorting generated artifacts
    into `good` and `bad` directories.
    """

    def __init__(self):
        super().__init__("repeat")
        self.subscribe_args(ArtifactSorterArgs(enable_iter_art=False))

    def add_parser_args(self, parser):
        parser.add_argument(
            "--until",
            required=True,
            help="Controls when to stop running. "
            "Choices are: ['good', 'bad', int]. 'good' will keep running until the first 'good' run. "
            "'bad' will run until the first 'bad' run. An integer can be specified to run a set number of iterations. ",
        )

    def run(self, args):
        try:
            until = int(args.until) - 1
        except:
            until = args.until
            if until not in ["good", "bad"]:
                G_LOGGER.critical("--until value must be an integer, 'good', or 'bad', but was: {:}".format(args.until))

        def stop(index, success):
            if until == "good":
                return success
            elif until == "bad":
                return not success

            return index >= until

        G_LOGGER.start("Starting iterations")

        num_passed = 0
        num_total = 0

        success = True
        MAX_COUNT = 100000  # We don't want to loop forever. This many iterations ought to be enough for anybody.
        for iteration in range(MAX_COUNT):
            G_LOGGER.start("RUNNING | Iteration {:}".format(iteration + 1))

            success = self.arg_groups[ArtifactSorterArgs].sort_artifacts(iteration + 1)

            num_total += 1
            if success:
                num_passed += 1

            if stop(iteration, success):
                break
        else:
            G_LOGGER.warning(
                "Maximum number of iterations reached: {:}.\n"
                "Iteration has been halted to prevent an infinite loop!".format(MAX_COUNT)
            )

        G_LOGGER.finish(
            "Finished {:} iteration(s) | Passed: {:}/{:} | Pass Rate: {:}%".format(
                iteration + 1, num_passed, num_total, float(num_passed) * 100 / float(num_total)
            )
        )
