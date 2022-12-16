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
import argparse
import contextlib
import copy
import os
import re
import shutil
import subprocess as sp
import time
from types import MappingProxyType

from polygraphy import config, util
from polygraphy.json import load_json, save_json, Decoder, Encoder
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs


class IterationContext:
    """
    Tracks per-iteration state and other contextual information during iterative debugging.
    """

    def __init__(self, state: dict, iteration_info: dict, success: bool):
        """
        Args:
            state (Dict[Any, Any]): The initial state of the context.
            iteration_info (Dict[Any, Any]): The initial iteration information of the context.
            success (bool): The initial success value of the context.
        """
        self.state = state
        """Tracks internal per-iteration state"""
        self.iteration_info = iteration_info
        """Tracks per-iteration state that should be exposed to the user, e.g. current iteration number"""
        self.success = success

    def freeze(self):
        """
        Freeze this context so that the `state` and `iteration_info` cannot be modified.
        """
        self.state = MappingProxyType(self.state)
        self.iteration_info = MappingProxyType(self.iteration_info)


@Encoder.register(IterationContext)
def encode(iter_context):
    return {
        "state": dict(iter_context.state),
        "iteration_info": dict(iter_context.iteration_info),
        "success": iter_context.success,
    }


@Decoder.register(IterationContext)
def decode(dct):
    return IterationContext(state=dct["state"], iteration_info=dct["iteration_info"], success=dct["success"])


class CheckCmdArgs(BaseArgs):
    """
    Pass/Fail Reporting: reporting pass/fail status during iterative debugging.
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--check",
            "--check-inference",
            dest="check",
            help="A command to check the model. When this is omitted, an interactive debugging session is started instead."
            "By default an exit status of 0 is treated as a 'pass' whereas any other exit status is treated as a 'fail'.",
            nargs=argparse.REMAINDER,
            default=None,
        )

        fail_codes = self.group.add_mutually_exclusive_group()
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

        self.group.add_argument(
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

        output_show = self.group.add_mutually_exclusive_group()
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

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            check (List[str]):
                    The check command.
            fail_codes (List[int]):
                    Exit status codes that represent failures.
            ignore_fail_codes (List[int]):
                    Exit status codes that should not be treated as failures.
            fail_regexes (List[re.SRE_Pattern]):
                    A list of compiled regular expressions that will cause the result of the command
                    to be treated as a failure if they are present in the output.
            show_output (bool):
                    Whether to show output for passing commands.
            hide_fail_output (bool):
                    Whether to hide output for failing commands.
        """
        self.check = args_util.get(args, "check")

        if self.check is None:
            G_LOGGER.start("Starting interactive debugging session since no `--check` command was provided")

        self.fail_codes = args_util.get(args, "fail_codes")
        self.ignore_fail_codes = args_util.get(args, "ignore_fail_codes")

        self.fail_regexes = None
        fail_regex = args_util.get(args, "fail_regex")
        if fail_regex is not None:
            self.fail_regexes = []
            for regex in fail_regex:
                self.fail_regexes.append(re.compile(regex))

        self.show_output = args_util.get(args, "show_output")
        self.hide_fail_output = args_util.get(args, "hide_fail_output")

    def run_check(self, iter_artifact_path):
        """
        Runs the check command and reports whether it succeeded.

        Args:
            iter_artifact_path (str):
                    The path to the intermediate artifact.
                    Use a value of None to indicate that there is no intermediate artifact.

        Returns:
            bool: Whether the check command passed.
        """

        # Interactive mode
        if self.check is None:

            def prompt_user(msg):
                res = input(f">>> {msg} ").lower()
                if len(res) < 1:
                    return None
                return res[0]

            prompt = (
                f"Did '{iter_artifact_path if iter_artifact_path is not None else 'this iteration'}' [p]ass or [f]ail?"
            )
            response = prompt_user(prompt)
            while response not in ["p", "f"]:
                response = prompt_user("Please choose either: 'p' or 'f':")
            return response == "p"

        def is_status_success(status):
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

        G_LOGGER.info(f"Running check command: {' '.join(self.check)}")
        status = sp.run(self.check, stdout=sp.PIPE, stderr=sp.PIPE)
        success = is_status_success(status)

        if self.show_output or (not success and not self.hide_fail_output):
            stderr_log_level = G_LOGGER.WARNING if success else G_LOGGER.ERROR
            G_LOGGER.info(f"========== CAPTURED STDOUT ==========\n{status.stdout.decode()}")
            G_LOGGER.log(
                f"========== CAPTURED STDERR ==========\n{status.stderr.decode()}",
                severity=stderr_log_level,
            )
        return success


class ArtifactSortArgs(BaseArgs):
    """
    Artifact Sorting: sorting artifacts into good/bad directories based on pass/fail status.
    """

    def __init__(self, allow_no_artifacts_warning=None):
        """
        Args:
            allow_no_artifacts_warning (bool):
                    Whether to issue a warning when no artifacts have been specified.
                    Defaults to True.
        """
        super().__init__()
        self._allow_no_artifacts_warning = util.default(allow_no_artifacts_warning, True)

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--artifacts",
            help="Path(s) of artifacts to sort. "
            "These will be moved into 'good' and 'bad' directories based on the exit status of "
            "the `--check` command and suffixed with an iteration number, timestamp and return code. ",
            nargs="+",
        )
        self.group.add_argument(
            "--art-dir",
            "--artifacts-dir",
            metavar="DIR",
            dest="artifacts_dir",
            help="The directory in which to move artifacts and sort them into 'good' and 'bad'. "
            "Defaults to a directory named `polygraphy_artifacts` in the current directory. ",
            default="polygraphy_artifacts",
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            artifacts (List[str]): The list of artifacts to manage.
            output_dir (str): Path to the output directory in which to sort artifacts.
            start_data (str): A string representing the current date.
            start_time (str): A string representing the time of parsing.
        """
        self.artifacts = util.default(args_util.get(args, "artifacts"), [])

        if not self.artifacts and self._allow_no_artifacts_warning:
            G_LOGGER.warning(
                "`--artifacts` was not specified; No artifacts will be stored during this run! "
                "Is this what you intended?"
            )

        self.output_dir = args_util.get(args, "artifacts_dir")

        self.start_date = time.strftime("%x").replace("/", "-")
        self.start_time = time.strftime("%X").replace(":", "-")

    def sort_artifacts(self, success, suffix):
        """
        Sorts all tracked artifacts into 'good' or 'bad' subdirectories based on the pass/fail status.

        Args:
            success (bool): Whether the iteration passed.
            suffix (str): The suffix to use for the new artifact name.
        """

        def move_artifacts(subdir):
            """
            Moves artifacts (args.artifacts) into the specified subdirectory or args.output and
            appends an index and timestamp. Creates parent directories as required.

            Args:
                subdir (str): The destination path as a subdirectory of args.output.
            """
            for art in self.artifacts:
                basename, ext = os.path.splitext(os.path.basename(art))
                name = f"{basename}_{self.start_date}_{self.start_time}{suffix}{ext}"
                dest = os.path.join(self.output_dir, subdir, name)

                if not os.path.exists(art):
                    G_LOGGER.error(
                        f"Artifact: {art} does not exist, skipping.\nWas the artifact supposed to be generated?"
                    )
                    continue

                if os.path.exists(dest):
                    G_LOGGER.error(
                        f"Destination path: {dest} already exists.\nRefusing to overwrite. This artifact will be skipped!"
                    )
                    continue

                G_LOGGER.info(f"Moving {art} to {dest}")

                util.makedirs(dest)
                shutil.move(art, dest)

        move_artifacts("good" if success else "bad")


class IterativeDebugArgs(BaseArgs):
    """
    Iterative Debugging: iteratively debugging.

    Depends on:

        - ArtifactSortArgs
        - CheckCmdArgs

    This argument group provides utilities for iterative debugging tools like `debug reduce` and `debug build`.
    Tools can provide callbacks that control what happens in each iteration and how to respond to success/failure.

    `IterativeDebugArgs` manages running these callbacks and optionally sorting artifacts generated in each iteration.
    """

    def __init__(
        self, allow_iter_art_opt=None, iter_art_opt_default=None, allow_until_opt=None, allow_debug_replay=None
    ):
        """
        Args:
            allow_iter_art_opt (bool):
                    Whether to allow specifying the intermediate artifact option.
                    Defaults to True.
            iter_art_opt_default (str):
                    The default name to use for the per-iteration intermediate artifact.
                    This is a required argument when ``allow_iter_art_opt`` is True.
            allow_until_opt (bool):
                    Whether to allow the ``--until`` option to control iterations.
                    If this is set to True, the ``iterate()`` method will *not* accept a ``advance_func`` argument.
                    Defaults to False.
            allow_debug_replay (bool):
                    Whether to allow saving/loading debug replays.
                    Defaults to True.
        """
        self._allow_iter_art_opt = util.default(allow_iter_art_opt, True)
        self._iter_art_opt_default = iter_art_opt_default

        if allow_iter_art_opt and not iter_art_opt_default:
            G_LOGGER.internal_error("Must provide iter_art_opt_default if intermediate artifact is enabled")

        self._allow_until_opt = util.default(allow_until_opt, False)
        self._allow_debug_replay = util.default(allow_debug_replay, True)

        if config.INTERNAL_CORRECTNESS_CHECKS:
            # `iterate()` has special requirements when it is called multiple times from a single tool.
            # When internal correctness checks are on, we'll use these extra members to check that those requirements
            # are satisfied.
            self.num_invocations = 0
            self.previous_suffixes = set()

    def allows_abbreviation_impl(self):
        return False

    def add_parser_args_impl(self):
        if self._allow_iter_art_opt:
            self.group.add_argument(
                "--iter-artifact",
                "--intermediate-artifact",
                dest="iter_artifact_path",
                help="Path to store the intermediate artifact from each iteration. "
                f"Defaults to: {self._iter_art_opt_default}",
                default=self._iter_art_opt_default,
            )
            self.group.add_argument(
                "--no-remove-intermediate",
                help="Do not remove the intermediate artifact between iterations. "
                "Subsequent iterations may still overwrite the artifact from previous iterations. "
                "This allows you to exit the tool early and still have access to the most recent intermediate artifact. ",
                action="store_false",
                dest="remove_intermediate",
                default=True,
            )

        self.group.add_argument(
            "--iter-info",
            "--iteration-info",
            help="Path to write a JSON file containing information about "
            "the current iteration. This will include an 'iteration' key whose value is the current iteration number. ",
            dest="iteration_info_path",
            default=None,
        )

        if self._allow_until_opt:
            self.group.add_argument(
                "--until",
                required=True,
                help="Controls when to stop running. "
                "Choices are: ['good', 'bad', int]. 'good' will keep running until the first 'good' run. "
                "'bad' will run until the first 'bad' run. An integer can be specified to run a set number of iterations. ",
            )

        if self._allow_debug_replay:
            self.group.add_argument(
                "--load-debug-replay",
                help="Path from which to load a debug replay."
                " A replay file includes information on the results of some or all iterations, allowing you to skip those iterations.",
                default=None,
            )

            self.group.add_argument(
                "--save-debug-replay",
                help="Path at which to save a debug replay, which includes information on the results of debugging iterations. "
                "The replay can be used with `--load-debug-replay` to skip iterations during subsequent debugging sessions. "
                "The replay is saved after the first iteration and overwritten with an updated replay during each iteration thereafter. "
                "This will also write a second replay file with a suffix of `_skip_current`, which is written before the iteration completes, "
                "and treats it as a failure. In cases where the iteration crashes, loading this replay file provides a means of skipping over the crash. "
                "Defaults to `polygraphy_debug_replay.json` in the current directory.",
                default="polygraphy_debug_replay.json",
            )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            iter_artifact_path (str):
                    Path to the per-iteration intermediate artifact.
            remove_intermediate (bool):
                    Whether to remove the intermediate artifact between iterations.
            iteration_info_path (str):
                    Path at which to write per-iteration metadata, like the current iteration index.
            until (Union[str, int]):
                    A string or integer indicating how long to iterate.
                    See the help message for ``--until`` for details.
            load_debug_replay (str):
                    Path from which to load a debug replay.
            save_debug_replay (str):
                    Path at which to save a debug replay.
        """
        self.iter_artifact_path = args_util.get(args, "iter_artifact_path")

        if self.iter_artifact_path and os.path.exists(self.iter_artifact_path):
            G_LOGGER.critical(
                f"{self.iter_artifact_path} already exists, refusing to overwrite.\n"
                "Please remove the file manually or specify a different path with `--intermediate-artifact`."
            )

        self.remove_intermediate = args_util.get(args, "remove_intermediate")
        self.iteration_info_path = args_util.get(args, "iteration_info_path")

        until = args_util.get(args, "until")
        if until is not None:
            try:
                until = int(until) - 1
            except:
                until = until
                if until not in ["good", "bad"]:
                    G_LOGGER.critical(f"--until value must be an integer, 'good', or 'bad', but was: {until}")

        self.until = until
        self.load_debug_replay = args_util.get(args, "load_debug_replay")
        self.save_debug_replay = args_util.get(args, "save_debug_replay")

    class SkipIteration(Exception):
        """
        Represents an exception indicating that the current iteration should be skipped.
        """

        def __init__(self, success):
            """
            Args:
                success (bool): Whether the skipped iteration should be treated as a success.
            """
            self.success = success

    def skip_iteration(self, success=None):
        """
        Indicate that the current iteration should be skipped prior to running checks.
        This may only be invoked from the ``make_iter_art_func()`` callback and may
        **only** be used if ``make_iter_art_func()`` does not return anything.

        Args:
            success (bool):
                    Whether the skipped iteration should be treated as a success.
                    Defaults to False.
        """
        success = util.default(success, False)
        raise IterativeDebugArgs.SkipIteration(success)

    def stop_iteration(self):
        """
        Indicate that ``iterate()`` should stop. This may be invoked from any of
        the callbacks provided to ``iterate()``.
        """
        raise StopIteration()

    def iterate(
        self,
        make_iter_art_func=None,
        advance_func=None,
        get_remaining_func=None,
        suffix=None,
        initial_debug_replay=None,
    ):
        """
        This method provides the overall functionality for iteratively debugging.
        It essentially does:
        ::

            while True:
                make_iter_art_func()
                run_check()
                sort_artifacts()
                advance_func()

        Args:
            make_iter_art_func (Callable[[IterationContext], None]):
                    A stateless callable that generates the per iteration intermediate artifact.
                    This callable may *write* state required by `advance_func` to `context.state` but not read from it.
                    It may also write inforamtion to `iteration_info` but must not modify any other members of the context.
                    This is critical to ensure that the debug replay mechanism works correctly.

                    This callback may call ``stop_iteration()`` or ``skip_iteration()`` to stop iteration completely,
                    or skip the current iteration respectively.

                    Any members set in the context must support JSON serialization.

            advance_func (Callable[[IterationContext], None]):
                    A callable which handles the success or failure of the check command and advances iteration.

                    The `context` includes at least the following information:
                        1. context.iteration_info["iteration"]
                                An integer representing the current iteration index.
                        2. context.success:
                                A boolean indicating whether the current iteration was successful.
                        3. context.state:
                                A dictionary that stores per-iteration state.

                        `advance_func` may *read* from the context but not write to it.

                    This callback may call ``stop_iteration()`` to stop iteration completely.
                    If ``make_iter_art_func`` requests the iteration to be skipped, this callback will *not* be called.

            get_remaining_func (Callable[[], int]):
                    A callable which takes no arguments and returns an estimate of the number of iterations remaining.

            suffix (str):
                    A suffix to use for replay files and sorted artifacts.
                    If ``iterate()`` is called multiple times by the same tool, the tool *must* provide a
                    unique suffix each time.

            initial_debug_replay (Dict[str, Any]):
                    The initial debug replay to use. This should come from the return value of a previous invocation of ``iterate()``.
                    Tools that call ``iterate()`` more than once *must* provide this parameter, and other tools *must not*.

        Returns:
            Dict[str, Any]: The debug replay file from this invocation.
        """

        if config.INTERNAL_CORRECTNESS_CHECKS:
            self.num_invocations += 1
            if self.num_invocations > 1:
                if suffix is None or initial_debug_replay is None:
                    G_LOGGER.internal_error(
                        "`iterate()` was called multiple times, but one or more required argument was note provided.\n"
                        f"Note: Required arguments are: [suffix (got '{suffix}'), initial_debug_replay (got '{initial_debug_replay}')]"
                    )

                if suffix in self.previous_suffixes:
                    G_LOGGER.internal_error(
                        f"The suffix argument to `iterate()` must be unique for any subsequent invocation, but got: '{suffix}'.\n"
                        f"Note: Previously provided suffixes were: {self.previous_suffixes}"
                    )

            self.previous_suffixes.add(suffix)

        def handle_until(context: IterationContext):
            index, success = context.iteration_info["iteration"], context.success
            if isinstance(self.until, str):
                if (self.until == "good" and success) or (self.until == "bad" and not success):
                    self.arg_groups[IterativeDebugArgs].stop_iteration()
            elif index >= self.until:
                self.arg_groups[IterativeDebugArgs].stop_iteration()

        if self._allow_until_opt:
            if advance_func is not None:
                G_LOGGER.internal_error(
                    "This method cannot accept a `advance_func` argument if `allow_until_opt` is True."
                )
            advance_func = handle_until

        def try_remove(path):
            def func():
                try:
                    os.remove(path)
                except:
                    G_LOGGER.warning(f"Could not clean up: {path}")

            return func

        debug_replay = {}
        if initial_debug_replay is not None:
            debug_replay = copy.copy(initial_debug_replay)
        elif self.load_debug_replay:
            # Only load from the disk the first time `iterate()`is used within a tool.
            # Subsequent invocations will provide `initial_debug_replay`.
            debug_replay = load_json(self.load_debug_replay, "debug replay")

        # We don't actually want to loop forever. This many iterations ought to be enough for anybody.
        MAX_COUNT = 100000
        num_passed = 0

        for index in range(MAX_COUNT):

            context = IterationContext(state={}, iteration_info={"iteration": index}, success=True)

            with contextlib.ExitStack() as stack, G_LOGGER.indent():
                remaining = get_remaining_func() if get_remaining_func is not None else None

                G_LOGGER.start(
                    f"RUNNING | Iteration {index + 1}{f' | Approximately {remaining} iteration(s) remaining' if remaining is not None else ''}"
                )

                def log_status(iter_success, start_time):
                    nonlocal num_passed

                    duration_in_sec = time.time() - start_time
                    if iter_success:
                        num_passed += 1
                        G_LOGGER.finish(f"PASSED | Iteration {index + 1} | Duration {duration_in_sec}s")
                    else:
                        G_LOGGER.error(f"FAILED | Iteration {index + 1} | Duration {duration_in_sec}s")

                # We must include the suffix in the debug replay key to disambiguate
                debug_replay_key = f"_N{index}" + (("_" + suffix) if suffix else "")

                start_time = time.time()
                if debug_replay_key in debug_replay:
                    context = debug_replay[debug_replay_key]
                    G_LOGGER.info(f"Loading iteration information from debug replay: success={context.success}")
                else:
                    # Ensure that the intermediate artifact will be removed at the end of the iteration if requested.
                    if self.iter_artifact_path and self.remove_intermediate:
                        stack.callback(try_remove(self.iter_artifact_path))

                    do_check = True
                    if make_iter_art_func is not None:
                        try:
                            make_iter_art_func(context)
                        except IterativeDebugArgs.SkipIteration as err:
                            context.success = err.success
                            do_check = False
                        except StopIteration:
                            break

                    if self.iteration_info_path:
                        save_json(context.iteration_info, self.iteration_info_path)
                        stack.callback(try_remove(self.iteration_info_path))

                    def save_replay(replay, description=None, suffix=None):
                        if self.save_debug_replay:
                            path = self.save_debug_replay
                            if suffix:
                                path = util.add_file_suffix(path, suffix)
                            save_json(replay, path, description)

                    # We save the replay twice - first with a status of FAIL, and then with the real status.
                    # This way, if `run_check()` causes a crash, we can treat the iteration as a failure
                    # and skip it to avoid the crash when we reload the replay file.
                    skip_current_context = copy.copy(context)
                    skip_current_context.success = False
                    debug_replay[debug_replay_key] = skip_current_context
                    save_replay(debug_replay, suffix="_skip_current")

                    if do_check:
                        context.success = self.arg_groups[CheckCmdArgs].run_check(self.iter_artifact_path)
                        self.arg_groups[ArtifactSortArgs].sort_artifacts(context.success, suffix=debug_replay_key)

                    debug_replay[debug_replay_key] = context
                    save_replay(debug_replay, "debug replay")

                log_status(context.success, start_time)

                # No further modifications should be made to the context
                context.freeze()

                try:
                    advance_func(context)
                except StopIteration:
                    break
        else:
            G_LOGGER.warning(
                f"Maximum number of iterations reached: {MAX_COUNT}.\n"
                "Iteration has been halted to prevent an infinite loop!"
            )

        num_total = index + 1
        G_LOGGER.finish(
            f"Finished {num_total} iteration(s) | Passed: {num_passed}/{num_total} | Pass Rate: {float(num_passed) * 100 / float(num_total)}%"
        )
        return debug_replay
