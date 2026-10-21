#!/usr/bin/env python3
"""
Polygraphy Plugin Autotuner - Basic CLI Demo

This script demonstrates the complete workflow of using Polygraphy Plugin Autotuner
to optimize ONNX models by replacing subgraphs with TensorRT plugins.

This is a cross-platform alternative to cli_demo.sh that works on Windows, Linux, and macOS.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from polygraphy.logger import G_LOGGER


def check_prerequisites():
    """Check basic prerequisites."""
    G_LOGGER.info("Checking prerequisites...")

    # Check Python version
    if sys.version_info < (3, 6):
        G_LOGGER.error("Python 3.6 or higher is required")
        return False

    # Note: Package dependencies (numpy, tensorrt, cuda-python, onnx) are handled
    # via lazy import with automatic installation when POLYGRAPHY_AUTOINSTALL_DEPS=1

    # Check if polygraphy is available
    try:
        result = subprocess.run(
            ["polygraphy", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            G_LOGGER.error("Polygraphy is not installed or not working correctly")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        G_LOGGER.error(
            "Polygraphy command not found. Install with: pip install polygraphy"
        )
        return False

    # Provide helpful message about automatic dependency installation
    G_LOGGER.info(
        "Note: Missing dependencies will be automatically installed if POLYGRAPHY_AUTOINSTALL_DEPS=1"
    )
    G_LOGGER.finish("Prerequisites check passed")
    return True


def create_test_model(num_groups, output_path):
    """Create test ONNX model."""
    G_LOGGER.info(f"Creating test model ({num_groups} Conv+ReLU groups)...")

    try:
        # Import the model creation function
        from create_test_model import create_conv_relu_model

        create_conv_relu_model(num_groups=num_groups, output_name=output_path)
        G_LOGGER.finish(f"Model created: {output_path}")
        return True
    except Exception as e:
        G_LOGGER.error(f"Failed to create model: {e}")
        return False


def run_optimization(input_model, output_model, report_path, patterns_dir, plugins_dir):
    """Run polygraphy plugin autotune."""
    G_LOGGER.info("Running optimization...")

    # Build the polygraphy command
    cmd = [
        "polygraphy",
        "plugin",
        "autotune",
        input_model,
        "--plugin-dir",
        patterns_dir,
        "--python-plugins",
        os.path.join(plugins_dir, "trt_conv_relu_plugin.py"),
        "--autotune-mode",
        "greedy",
        "--max-combinations",
        "1000",
        "--threshold",
        "0.1",
        "-o",
        output_model,
        "--report",
        report_path,
    ]

    # Print the actual command being executed
    G_LOGGER.info("")
    G_LOGGER.info("Executing Polygraphy command:")
    G_LOGGER.info(" ".join(cmd))
    G_LOGGER.info("")

    try:
        subprocess.run(cmd, check=True)
        G_LOGGER.finish("Optimization completed")
        G_LOGGER.info(f"  Output: {output_model}")
        G_LOGGER.info(f"  Report: {report_path}")
        return True
    except subprocess.CalledProcessError as e:
        G_LOGGER.error(f"Optimization failed with return code {e.returncode}")
        return False
    except Exception as e:
        G_LOGGER.error(f"Unexpected error during optimization: {e}")
        return False


def display_results(report_path):
    """Display optimization results."""
    G_LOGGER.info("")
    G_LOGGER.info("=== Results Summary ===")
    G_LOGGER.info("")

    if os.path.exists(report_path):
        try:
            with open(report_path, "r") as f:
                lines = f.readlines()
                # Display first 15 lines of the report
                for line in lines[:15]:
                    print(line.rstrip())
        except Exception as e:
            G_LOGGER.warning(f"Could not read report file: {e}")
    else:
        G_LOGGER.warning(f"Report file not found: {report_path}")


def cleanup(files_to_remove):
    """Clean up generated files."""
    G_LOGGER.warning("Cleaning up temporary files...")

    # Ask user for confirmation
    try:
        response = input("Remove generated files? (y/N): ").strip().lower()
        if response in ["y", "yes"]:
            for file_path in files_to_remove:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        G_LOGGER.info(f"Removed: {file_path}")
                    except Exception as e:
                        G_LOGGER.warning(f"Could not remove {file_path}: {e}")
            G_LOGGER.finish("Cleanup completed")
        else:
            G_LOGGER.info("Files preserved")
    except KeyboardInterrupt:
        print()
        G_LOGGER.info("Cleanup cancelled")


def main():
    parser = argparse.ArgumentParser(
        description="Polygraphy Plugin Autotuner - Basic CLI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_demo.py
  python cli_demo.py --groups 5
  python cli_demo.py --no-cleanup

Environment Variables:
  NUM_GROUPS    Number of Conv+ReLU groups (default: 2)
        """,
    )

    parser.add_argument(
        "--groups",
        type=int,
        default=int(os.environ.get("NUM_GROUPS", 2)),
        help="Number of Conv+ReLU groups (default: 2)",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true", help="Skip cleanup prompt at the end"
    )

    args = parser.parse_args()

    # Configuration
    script_dir = Path(__file__).parent
    input_model = "conv_relu_test_model.onnx"
    optimized_model = "optimized_model.onnx"
    optimization_report = "optimization_report.txt"
    patterns_dir = str(script_dir.parent / "patterns")
    plugins_dir = str(script_dir.parent / "plugins")

    # Print header
    print("=" * 72)
    print("       Polygraphy Plugin Autotuner - Basic CLI Demo")
    print("=" * 72)
    print()

    # Run the demo
    if not check_prerequisites():
        return 1

    if not create_test_model(args.groups, input_model):
        return 1

    if not run_optimization(
        input_model, optimized_model, optimization_report, patterns_dir, plugins_dir
    ):
        return 1

    display_results(optimization_report)

    G_LOGGER.info("")
    G_LOGGER.finish("Demo completed!")
    G_LOGGER.info("")

    # Offer cleanup
    if not args.no_cleanup:
        files_to_cleanup = [
            input_model,
            optimized_model,
            optimization_report,
            "optimized_model.onnx.report.txt",
            "config.yaml",
        ]
        cleanup(files_to_cleanup)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        G_LOGGER.warning("Demo interrupted by user")
        sys.exit(130)
