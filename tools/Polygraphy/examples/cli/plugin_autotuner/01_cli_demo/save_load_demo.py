#!/usr/bin/env python3
"""
Polygraphy Plugin Autotuner - Save/Load Config Demo

Demonstrates the workflow of:
1) Matching patterns and saving configuration (without optimization)
2) Loading saved configuration and running optimization

This is a cross-platform alternative to save_load_demo.sh that works on Windows, Linux, and macOS.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from polygraphy.logger import G_LOGGER


def create_test_model_if_needed(model_path, num_groups=5):
    """Create test model if it doesn't exist."""
    if os.path.exists(model_path):
        G_LOGGER.info(f"Using existing model: {model_path}")
        return True

    G_LOGGER.info(f"Creating test model ({num_groups} Conv+ReLU groups)...")
    try:
        from create_test_model import create_conv_relu_model

        create_conv_relu_model(num_groups=num_groups, output_name=model_path)
        G_LOGGER.finish(f"Model created: {model_path}")
        return True
    except Exception as e:
        G_LOGGER.error(f"Failed to create model: {e}")
        return False


def run_match_patterns(input_model, config_path, patterns_dir):
    """Step 1: Match patterns and save configuration."""
    G_LOGGER.info("")
    G_LOGGER.info("Step 1: Matching patterns and saving configuration...")

    cmd = [
        "polygraphy",
        "plugin",
        "match",
        input_model,
        "--plugin-dir",
        patterns_dir,
        "-o",
        config_path,
    ]

    # Print the actual command being executed
    G_LOGGER.info("")
    G_LOGGER.info("Executing Polygraphy command:")
    G_LOGGER.info(" ".join(cmd))
    G_LOGGER.info("")

    try:
        subprocess.run(cmd, check=True)

        if os.path.exists(config_path):
            G_LOGGER.finish(f"Config saved: {config_path}")
            G_LOGGER.info(
                "View the config file to see all matched patterns and their details"
            )
            return True
        else:
            G_LOGGER.error("Configuration was not generated")
            return False
    except subprocess.CalledProcessError as e:
        G_LOGGER.error(f"Pattern matching failed with return code {e.returncode}")
        return False
    except Exception as e:
        G_LOGGER.error(f"Unexpected error during pattern matching: {e}")
        return False


def run_optimization_with_config(
    input_model, output_model, report_path, config_path, patterns_dir, plugins_dir
):
    """Step 2: Load configuration and run optimization."""
    G_LOGGER.info("")
    G_LOGGER.info("Step 2: Loading configuration and running optimization...")

    cmd = [
        "polygraphy",
        "plugin",
        "autotune",
        input_model,
        "--plugin-dir",
        patterns_dir,
        "--python-plugins",
        os.path.join(plugins_dir, "trt_conv_relu_plugin.py"),
        "--load-config-yaml",
        config_path,
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

        if os.path.exists(output_model):
            G_LOGGER.finish(f"Optimized model: {output_model}")

            if os.path.exists(report_path):
                G_LOGGER.finish(f"Report: {report_path}")
                G_LOGGER.info("")

                # Display first 15 lines of the report
                try:
                    with open(report_path, "r") as f:
                        lines = f.readlines()
                        for line in lines[:15]:
                            print(line.rstrip())
                except Exception as e:
                    G_LOGGER.warning(f"Could not read report file: {e}")

            return True
        else:
            G_LOGGER.error("Optimization failed")
            return False
    except subprocess.CalledProcessError as e:
        G_LOGGER.error(f"Optimization failed with return code {e.returncode}")
        return False
    except Exception as e:
        G_LOGGER.error(f"Unexpected error during optimization: {e}")
        return False


def cleanup(files_to_remove):
    """Clean up generated files."""
    G_LOGGER.warning("Cleaning up temporary files...")

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
            G_LOGGER.info("Files preserved in current directory")
    except KeyboardInterrupt:
        print()
        G_LOGGER.info("Cleanup cancelled")


def main():
    parser = argparse.ArgumentParser(
        description="Polygraphy Plugin Autotuner - Save/Load Config Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python save_load_demo.py               # Run both steps
  python save_load_demo.py --step save   # Only run pattern matching
  python save_load_demo.py --step load   # Only run optimization
  python save_load_demo.py --no-cleanup  # Skip cleanup prompt

This script demonstrates a two-step workflow:
  Step 1 (save): Match patterns and save configuration YAML
  Step 2 (load): Load configuration and run optimization
        """,
    )

    parser.add_argument(
        "--step",
        choices=["save", "load"],
        help="Run only the specified step (default: run both)",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true", help="Skip cleanup prompt at the end"
    )

    args = parser.parse_args()

    # Configuration
    script_dir = Path(__file__).parent
    input_model = "conv_relu_test_model.onnx"
    matches_config = "matches.yaml"
    optimized_model = "optimized_from_config.onnx"
    optimization_report = "optimized_from_config.onnx.report.txt"
    patterns_dir = str(script_dir.parent / "patterns")
    plugins_dir = str(script_dir.parent / "plugins")

    # Print header
    print("=" * 72)
    print("     Polygraphy Plugin Autotuner - Save/Load Config Demo")
    print("=" * 72)
    print()

    # Ensure test model exists
    if not create_test_model_if_needed(input_model):
        return 1

    # Step 1: Match patterns and save configuration
    if args.step is None or args.step == "save":
        if not run_match_patterns(input_model, matches_config, patterns_dir):
            return 1

    # Step 2: Load configuration and run optimization
    if args.step is None or args.step == "load":
        if not os.path.exists(matches_config):
            G_LOGGER.error(f"Config file not found: {matches_config}")
            G_LOGGER.error("Run with '--step save' first to generate the config")
            return 1

        if not run_optimization_with_config(
            input_model,
            optimized_model,
            optimization_report,
            matches_config,
            patterns_dir,
            plugins_dir,
        ):
            return 1

    # Summary
    G_LOGGER.info("")
    G_LOGGER.finish("Demo completed!")
    G_LOGGER.info("")

    # Offer cleanup
    if not args.no_cleanup:
        files_to_cleanup = [
            input_model,
            matches_config,
            optimized_model,
            optimization_report,
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
