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


@mod.export()
def main():
    """
    The Polygraphy CLI Toolkit

    Includes various subtools that can assist with prototyping and
    debugging inference with deep learning models. See the help output
    for details.
    """
    import argparse
    import sys

    import polygraphy
    from polygraphy.exception import PolygraphyException
    from polygraphy.logger import G_LOGGER
    from polygraphy.tools.registry import TOOL_REGISTRY

    parser = argparse.ArgumentParser(
        description="Polygraphy: A Deep Learning Debugging Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--version", action="version", version=G_LOGGER._str_from_module_info(polygraphy, name="Polygraphy")
    )

    subparsers = parser.add_subparsers(title="Tools", dest="tools")
    subparsers.required = True

    for tool in TOOL_REGISTRY:
        tool.setup_parser(subparsers)

    args, unknown = parser.parse_known_args()

    if unknown:
        G_LOGGER.error(f"Unrecognized Options: {unknown}")
        return 1

    G_LOGGER.verbose(f"Running Command: {' '.join(sys.argv)}")

    try:
        status = args.subcommand(args)
    except PolygraphyException:
        # `PolygraphyException`s indicate user error, so we need not display the stack trace.
        status = 1

    return status
