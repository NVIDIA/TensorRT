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
import argparse
import contextlib
import os
import re
import shutil
import subprocess as sp
import time

from polygraphy import util
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs


class ArtifactSorterArgs(BaseArgs):
    def __init__(self, iter_art_default=None, prefer_artifacts=True, enable_iter_art=True):
        assert (
            iter_art_default or not enable_iter_art
        ), "Must provide iter_art_default if intermediate artifact is enabled"
        super().__init__(disable_abbrev=True)
        self._iter_art_default = iter_art_default
        self._prefer_artifacts = prefer_artifacts
        self._enable_iter_art = enable_iter_art

    def add_to_parser(self, parser):
        artifact_sorter_args = parser.add_argument_group("Artifact Sorting", "Options for sorting artifacts")
        artifact_sorter_args.add_argument(
            "--artifacts",
            help="Path(s) of artifacts to sort. "
            "These will be moved into 'good' and 'bad' directories based on the exit status of "
            "the `--check` command and suffixed with an iteration number, timestamp and return code. ",
            nargs="+",
        )
        artifact_sorter_args.add_argument(
            "--art-dir",
            "--artifacts-dir",
            metavar="DIR",
            dest="artifacts_dir",
            help="The directory in which to move artifacts and sort them into 'good' and 'bad'. ",
        )

        artifact_sorter_args.add_argument(
            "--check",
            "--check-inference",
            dest="check",
            help="A command to check the model. "
            "The command should return an exit status of 0 for the run to be considered 'good'. "
            "Non-zero exit statuses are treated as 'bad' runs.",
            required=True,
            nargs=argparse.REMAINDER,
        )

        fail_codes = artifact_sorter_args.add_mutually_exclusive_group()
        fail_codes.add_argument(
            "--fail-code",
            "--fail-returncode",
            dest="fail_codes",
            help="The return code(s) from the --check command to count as failures. "
            "If this is provided, any other return code will be counted as a success. ",
            nargs="+",
            default=None,
            type=int,
        )
        fail_codes.add_argument(
            "--ignore-fail-code",
            "--ignore-fail-returncode",
            dest="ignore_fail_codes",
            help="The return code(s) from the --check command to ignore as failures. ",
            nargs="+",
            default=None,
            type=int,
        )

        artifact_sorter_args.add_argument(
            "--fail-regex",
            dest="fail_regex",
            help="Regular expression denoting an error in the check command's output. The command "
            "is only considered a failure if a matching string is found in the command's output. "
            "This can be useful to distinguish among multiple types of failures. "
            "Can be specified multiple times to match different regular expressions, in which case any match counts as a failure. "
            "When combined with --fail-code, only iterations whose return code is considered a failure are "
            "checked for regular expressions.",
            default=None,
            nargs="+",
        )

        output_show = artifact_sorter_args.add_mutually_exclusive_group()
        output_show.add_argument(
            "--show-output",
            help="Show output from the --check command even for passing iterations. "
            "By default, output from passing iterations is captured. ",
            action="store_true",
        )
        output_show.add_argument(
            "--hide-fail-output",
            help="Suppress output from the --check command for failing iterations. "
            "By default, output from failing iterations is displayed. ",
            action="store_true",
        )

        if self._enable_iter_art:
            artifact_sorter_args.add_argument(
                "--iter-artifact",
                "--intermediate-artifact",
                dest="iter_artifact",
                help="Path to store the intermediate artifact from each iteration. "
                "Defaults to: {:}".format(self._iter_art_default),
                default=self._iter_art_default,
            )
            artifact_sorter_args.add_argument(
                "--no-remove-intermediate",
                help="Do not remove the intermediate artifact between iterations. "
                "This allows you to exit the tool early and still have access to the intermediate artifact. ",
                action="store_false",
                dest="remove_intermediate",
            )

        artifact_sorter_args.add_argument(
            "--iter-info",
            "--iteration-info",
            help="Path to write a JSON file containing information about "
            "the current iteration. This will include an 'iteration' key specifying the current iteration. ",
            dest="iteration_info",
            default=None,
        )

    def parse(self, args):
        self.iter_artifact = args_util.get(args, "iter_artifact")

        if self.iter_artifact and os.path.exists(self.iter_artifact):
            G_LOGGER.critical(
                "{:} already exists, refusing to overwrite.\n"
                "Please specify a different path for the intermediate artifact with "
                "--intermediate-artifact".format(self.iter_artifact)
            )

        self.artifacts = util.default(args_util.get(args, "artifacts"), [])
        self.output = args_util.get(args, "artifacts_dir")
        self.show_output = args_util.get(args, "show_output")
        self.hide_fail_output = args_util.get(args, "hide_fail_output")
        self.remove_intermediate = args_util.get(args, "remove_intermediate")
        self.fail_codes = args_util.get(args, "fail_codes")
        self.ignore_fail_codes = args_util.get(args, "ignore_fail_codes")

        self.fail_regexes = None
        fail_regex = args_util.get(args, "fail_regex")
        if fail_regex is not None:
            self.fail_regexes = []
            for regex in fail_regex:
                self.fail_regexes.append(re.compile(regex))

        if self.artifacts and not self.output:
            G_LOGGER.critical(
                "An output directory must be specified if artifacts are enabled! "
                "Note: Artifacts specified were: {:}".format(self.artifacts)
            )

        if not self.artifacts and self._prefer_artifacts:
            G_LOGGER.warning(
                "`--artifacts` was not specified; No artifacts will be stored during this run! "
                "Is this what you intended?"
            )

        self.iteration_info = args_util.get(args, "iteration_info")

        self.check = args_util.get(args, "check")

        self.start_date = time.strftime("%x").replace("/", "-")
        self.start_time = time.strftime("%X").replace(":", "-")

    def sort_artifacts(self, iteration, suffix=None):
        """
        Run the check command and move artifacts into the correct subdirectory.

        Args:
            iteration (int):
                    The current iteration index. This is used to name artifacts
                    and display logging messages.

            suffix (str):
                    A custom suffix to add to the artifact prior to moving it.
                    This will be applied in addition to the default suffix.
        Returns:
            bool: True if the command succeeded, False otherwise.
        """

        def move_artifacts(subdir, returncode):
            """
            Moves artifacts (args.artifacts) into the specified subdirectory or args.output and
            appends an index and timestamp. Creates parent directories as required.

            Args:
                subdir (str): The destination path as a subdirectory of args.output.
                index (int): The iteration index.
            """
            for art in self.artifacts:
                basename, ext = os.path.splitext(os.path.basename(art))
                if suffix:
                    basename += suffix
                name = "{:}_{:}_{:}_N{:}_ret{:}{:}".format(
                    basename, self.start_date, self.start_time, iteration, returncode, ext
                )
                dest = os.path.join(self.output, subdir, name)

                if not os.path.exists(art):
                    G_LOGGER.error(
                        "Artifact: {:} does not exist, skipping.\n"
                        "Was the artifact supposed to be generated?".format(art)
                    )
                    continue

                if os.path.exists(dest):
                    G_LOGGER.error(
                        "Destination path: {:} already exists.\n"
                        "Refusing to overwrite. This artifact will be skipped!".format(dest)
                    )
                    continue

                G_LOGGER.info("Moving {:} to {:}".format(art, dest))

                dir_path = os.path.dirname(dest)
                if dir_path:
                    dir_path = os.path.realpath(dir_path)
                    os.makedirs(dir_path, exist_ok=True)
                shutil.move(art, dest)

        def try_remove(path):
            def func():
                try:
                    os.remove(path)
                except:
                    G_LOGGER.verbose("Could not remove: {:}".format(path))

            return func

        def is_success(status):
            if self.ignore_fail_codes and status.returncode in self.ignore_fail_codes:
                return True

            has_fail_regex = None
            if self.fail_regexes is not None:
                output = status.stdout.decode() + status.stderr.decode()
                has_fail_regex = any(regex.search(output) is not None for regex in self.fail_regexes)

            if self.fail_codes is not None:
                # If a fail-code is specified, then we should also check has_fail_regex if provided.
                failed = status.returncode in self.fail_codes
                if has_fail_regex is not None:
                    failed &= has_fail_regex
            else:
                # If a fail-code is not specified, we should trigger failures even on 0-status
                # if the fail regex is found.
                failed = status.returncode != 0 if has_fail_regex is None else has_fail_regex
            return not failed

        with contextlib.ExitStack() as stack, G_LOGGER.indent():
            if self.iter_artifact and self.remove_intermediate:
                stack.callback(try_remove(self.iter_artifact))

            if self.iteration_info:
                util.save_json({"iteration": iteration}, self.iteration_info)
                stack.callback(try_remove(self.iteration_info))

            G_LOGGER.info("Running check command: {:}".format(" ".join(self.check)))
            status = sp.run(self.check, stdout=sp.PIPE, stderr=sp.PIPE)
            success = is_success(status)

            if self.show_output or (not success and not self.hide_fail_output):
                stderr_log_level = G_LOGGER.WARNING if success else G_LOGGER.ERROR
                G_LOGGER.info("========== CAPTURED STDOUT ==========\n{:}".format(status.stdout.decode()))
                G_LOGGER.log(
                    "========== CAPTURED STDERR ==========\n{:}".format(status.stderr.decode()),
                    severity=stderr_log_level,
                )

            if success:
                move_artifacts("good", status.returncode)
                G_LOGGER.finish("PASSED | Iteration {:}".format(iteration))
                return True
            else:
                move_artifacts("bad", status.returncode)
                G_LOGGER.error("FAILED | Iteration {:}".format(iteration))
                return False
