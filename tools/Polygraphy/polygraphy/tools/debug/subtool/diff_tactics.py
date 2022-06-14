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
import glob
import os
from collections import OrderedDict, defaultdict

from polygraphy import util
from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.base import Tool

algorithm_selector = mod.lazy_import("polygraphy.backend.trt.algorithm_selector")


class DiffTactics(Tool):
    """
    Determine potentially bad tactics given sets of good and bad tactic
    replay files.
    """

    def __init__(self):
        super().__init__("diff-tactics")

    def add_parser_args(self, parser):
        parser.add_argument(
            "--dir",
            help="A directory containing good and bad tactic replay files. "
            "By default, this tool will search for files in directories called 'good' and 'bad'",
            default="",
        )
        parser.add_argument(
            "--good",
            help="A directory containing good tactic replay files or a single good tactic replay file. ",
            default=None,
        )
        parser.add_argument(
            "--bad",
            help="A directory containing bad tactic replay files or a single bad tactic replay file. ",
            default=None,
        )

    def run(self, args):
        if args.dir is None and (args.good is None or args.bad is None):
            G_LOGGER.critical("Either `--dir`, or both `--good` and `--bad` must be specified.")

        def load_tactics(dirpath):
            """
            Load all tactic replays from the specified directory into a single dictionary.

            Args:
                dirpath (str): Directory containing zero or more tactic replay files.

            Returns:
                dict[str, Set[polygraphy.backend.trt.algorithm_selector.Algorithm]]:
                        Maps layer names to the set of algorithms present in the tactic replays.
            """

            def try_load_replay(path):
                try:
                    return algorithm_selector.TacticReplayData.load(path)
                except:
                    return None

            tactics = defaultdict(set)
            replay_paths = []
            search_paths = (
                glob.iglob(os.path.join(dirpath, "**"), recursive=True) if os.path.isdir(dirpath) else [dirpath]
            )
            for path in search_paths:
                replay = try_load_replay(path)
                if replay is None:
                    G_LOGGER.verbose("{:} does not look like a tactic replay file, skipping.".format(path))
                    continue

                replay_paths.append(path)
                for name, algo in replay.items():
                    tactics[name].add(algo)
            return tactics, replay_paths

        good_dir = util.default(args.good, os.path.join(args.dir, "good"))
        good_tactics, good_paths = load_tactics(good_dir)
        G_LOGGER.info("Loaded {:} good tactic replays.".format(len(good_paths)))
        G_LOGGER.verbose("Good tactic replays: {:}".format(good_paths))

        bad_dir = util.default(args.bad, os.path.join(args.dir, "bad"))
        bad_tactics, bad_paths = load_tactics(bad_dir)
        G_LOGGER.info("Loaded {:} bad tactic replays.".format(len(bad_paths)))
        G_LOGGER.verbose("Bad tactic replays: {:}".format(bad_paths))

        # Walk bad tactics and remove all the known good tactics.
        potential_bad_tactics = OrderedDict()
        for name, algo_set in bad_tactics.items():
            if name in good_tactics:
                algo_set -= good_tactics[name]

            if algo_set:
                potential_bad_tactics[name] = algo_set

        if potential_bad_tactics:
            G_LOGGER.info("Found potentially bad tactics:")
            for name, algo_set in potential_bad_tactics.items():
                algo_set_str = list(map(str, algo_set))
                G_LOGGER.info("Layer: {:}\n\tAlgorithms: {:}".format(name, algo_set_str))
        else:
            G_LOGGER.info("Could not determine potentially bad tactics. Try generating more tactic replay files?")
