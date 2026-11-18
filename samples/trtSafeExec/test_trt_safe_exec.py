# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI regression tests requiring a GPU and a compatible prebuilt safety engine.

Set TRTEXEC_SAFE_TEST_BINARY and TRTEXEC_SAFE_TEST_ENGINE to absolute paths.
Run: python3 tools/trtSafeExec/test_trt_safe_exec.py
"""

import json
import os
import sys
import subprocess
import tempfile
import unittest


class MemoryDiagnosticsCli(unittest.TestCase):
    """Exercise cycle parsing and initialization logging through the real executable."""

    @classmethod
    def setUpClass(cls):
        """Read paths supplied by the build/test environment."""
        cls.binary = os.environ.get("TRTEXEC_SAFE_TEST_BINARY")
        cls.engine = os.environ.get("TRTEXEC_SAFE_TEST_ENGINE")
        if not cls.binary or not cls.engine:
            raise unittest.SkipTest("Set TRTEXEC_SAFE_TEST_BINARY and TRTEXEC_SAFE_TEST_ENGINE")

    def run_tool(self, *options, environment=None):
        """Use a short inference run and a writable directory for runtime error logs."""
        args = [self.binary, f"--loadEngine={self.engine}", "--iterations=1", "--avgRuns=1",
                "--warmUp=1", "--duration=0", "--sleepTime=0", "--noCudaGraph", *options]
        with tempfile.TemporaryDirectory() as directory:
            return subprocess.run(args, cwd=directory, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, timeout=300, env=environment)

    def test_repeated_cycles_report_initialization_once(self):
        """Later graph lifetimes must not report elapsed time from the first cycle."""
        result = self.run_tool("--reportMemoryUsage", "--memoryCycles=3")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("Memory lifecycle cycle "), 3)
        self.assertEqual(result.stdout.count("TensorRT init time is "), 1)
        self.assertRegex(result.stdout, r"TensorRT init time is \d+(?:\.\d+)? ms\.")
        self.assertEqual(result.stdout.count("Tracked allocation cleanup check: PASSED"), 4)

    def test_default_reports_one_cycle_and_initialization(self):
        """The default remains one normal graph lifetime and one profiling run."""
        result = self.run_tool("--reportMemoryUsage")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("Memory lifecycle cycle "), 1)
        self.assertEqual(result.stdout.count("TensorRT init time is "), 1)
        self.assertRegex(result.stdout, r"TensorRT init time is \d+(?:\.\d+)? ms\.")
        self.assertEqual(result.stdout.count("Tracked allocation cleanup check: PASSED"), 2)

    def test_debug_and_memory_reporting_are_independent(self):
        """Memory accounting and runtime debug logging can be selected separately."""
        for debug in (False, True):
            for report_memory in (False, True):
                with self.subTest(debug=debug, report_memory=report_memory):
                    options = (["--debug"] if debug else [])
                    options += (["--reportMemoryUsage"] if report_memory else [])
                    result = self.run_tool(*options)
                    self.assertEqual(result.returncode, 0, result.stdout)
                    for message in ("Memory lifecycle cycle ", "Memory lifecycle: ",
                                    "Tracked allocation peak:", "Tracked allocation cleanup check:"):
                        self.assertEqual(message in result.stdout, report_memory, result.stdout)

    def test_invalid_cycle_counts_are_rejected(self):
        """Reject malformed and out-of-range counts without starting graph cycles."""
        for value in ("0", "-1", "1001", "3x", "NaN", "999999999999999999999999"):
            with self.subTest(value=value):
                result = self.run_tool("--reportMemoryUsage", f"--memoryCycles={value}")
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("memoryCycles must be an integer in [1, 1000].", result.stdout)
                self.assertNotIn("Memory lifecycle cycle ", result.stdout)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Automatic sanitizer launch is Linux-only")
    def test_linux_memory_reporting_runs_memcheck(self):
        """The real sanitizer must run once and report its own error/leak summaries."""
        result = self.run_tool("--reportMemoryUsage")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("Launching Compute Sanitizer memcheck"), 1)
        self.assertIn("ERROR SUMMARY: 0 errors", result.stdout)
        self.assertIn("LEAK SUMMARY: 0 bytes leaked", result.stdout)
        self.assertIn("Tracked allocation cleanup check: PASSED", result.stdout)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Automatic sanitizer launch is Linux-only")
    def test_missing_sanitizer_fails_but_help_does_not_launch(self):
        """Missing tooling must never produce a misleading successful memory check."""
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ, PATH=directory)
            result = self.run_tool("--reportMemoryUsage", environment=environment)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("Cannot launch compute-sanitizer:", result.stdout)
            self.assertNotIn("Tracked allocation cleanup check: PASSED", result.stdout)
            result = self.run_tool("--reportMemoryUsage", "--help", environment=environment)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertNotIn("Launching Compute Sanitizer", result.stdout)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Automatic sanitizer launch is Linux-only")
    def test_sanitizer_arguments_and_exit_status(self):
        """Forward arguments without shell interpretation and preserve sanitizer failures."""
        with tempfile.TemporaryDirectory() as directory:
            shim = os.path.join(directory, "compute-sanitizer")
            with open(shim, "w", encoding="utf-8") as script:
                script.write(f"#!{sys.executable}\nimport json, sys\n"
                             "print('SANITIZER_ARGS:' + json.dumps(sys.argv[1:]))\nsys.exit(42)\n")
            os.chmod(shim, 0o700)
            result = self.run_tool("--reportMemoryUsage", "--loadInputs=tensor:path with spaces;$value",
                                   environment=dict(os.environ, PATH=directory))
            self.assertEqual(result.returncode, 42, result.stdout)
            line = next(line for line in result.stdout.splitlines() if line.startswith("SANITIZER_ARGS:"))
            arguments = json.loads(line.removeprefix("SANITIZER_ARGS:"))
            self.assertEqual(arguments[:8], ["--tool", "memcheck", "--leak-check", "full",
                                            "--error-exitcode", "99", "--", self.binary])
            self.assertIn("--loadInputs=tensor:path with spaces;$value", arguments)

    def test_cycle_boundaries_are_accepted(self):
        """Validate both bounds through help without executing a thousand cycles."""
        for value in ("1", "1000"):
            with self.subTest(value=value):
                result = self.run_tool("--reportMemoryUsage", f"--memoryCycles={value}", "--help")
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertNotIn("memoryCycles must be an integer", result.stdout)

    def test_cycle_option_requires_memory_reporting(self):
        """An explicitly supplied cycle option requires diagnostics even at one cycle."""
        for debug in (False, True):
            for value in ("1", "2"):
                with self.subTest(debug=debug, value=value):
                    options = ["--debug"] if debug else []
                    result = self.run_tool(*options, f"--memoryCycles={value}")
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("--memoryCycles requires --reportMemoryUsage.", result.stdout)


if __name__ == "__main__":
    unittest.main()
