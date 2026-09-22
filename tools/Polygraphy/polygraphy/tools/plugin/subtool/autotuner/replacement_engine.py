"""
Pattern-based replacement engine that wraps existing Polygraphy plugin tools
(`match` and `replace`) to find and apply plugin subgraph replacements.
"""

import glob
import os
from typing import Dict, List, Any, Optional

from polygraphy import mod, util
from polygraphy.logger import G_LOGGER

yaml = mod.lazy_import("yaml", pkg_name="pyyaml")


class _ToolArgs:
    """Mock args object for internal tool calls."""

    def __init__(self, **kwargs):
        self.arg_groups = {}
        for key, value in kwargs.items():
            setattr(self, key, value)


class ReplacementEngine:

    def __init__(self):
        self.plugin_dir: Optional[str] = None
        self.plugins: Dict[str, Dict[str, str]] = {}
        self.plugin_subgraphs: Dict[str, List[Dict[str, Any]]] = {}

    def load_plugins(
        self,
        plugin_dir: str,
        include_list: Optional[List[str]] = None,
        exclude_list: Optional[List[str]] = None,
    ):
        from polygraphy.tools.plugin.subtool.plugin_base import PluginBase

        abs_plugin_dir = os.path.abspath(plugin_dir)
        if not os.path.exists(abs_plugin_dir):
            G_LOGGER.critical(f"Plugin directory does not exist: {plugin_dir}")

        self.plugin_dir = abs_plugin_dir
        base = PluginBase(list_plugins=False)
        full_pattern = os.path.join(abs_plugin_dir, "*", base.GRAPH_PATTERN_FILE_NAME)

        plugin_set = {
            os.path.basename(os.path.dirname(x))
            for x in glob.glob(pathname=full_pattern, recursive=False)
        }

        if include_list:
            plugin_set.intersection_update(set(include_list))
        if exclude_list:
            plugin_set.difference_update(set(exclude_list))

        for name in plugin_set:
            pattern_file = os.path.join(
                abs_plugin_dir, name, base.GRAPH_PATTERN_FILE_NAME
            )
            if os.path.exists(pattern_file):
                self.plugins[name] = {"pattern_file": pattern_file, "name": name}

        G_LOGGER.info(f"Loaded {len(self.plugins)} plugin(s) from {plugin_dir}")

    def _load_config_from_yaml(self, config_path: str):
        """Load plugin configuration from YAML file."""
        config_content = util.load_file(
            config_path, mode="r", description="plugin config"
        )
        plugin_docs = list(yaml.safe_load_all(config_content))

        self.plugin_subgraphs = {}
        for plugin_data in plugin_docs:
            plugin_name = plugin_data.get("name", "unknown")
            instances = plugin_data.get("instances", [])
            self.plugin_subgraphs[plugin_name] = []
            for i, instance in enumerate(instances):
                self.plugin_subgraphs[plugin_name].append(
                    {
                        "index": i,
                        "inputs": instance.get("inputs", []),
                        "outputs": instance.get("outputs", []),
                        "attributes": instance.get("attributes", {}),
                        "plugin_name": plugin_name,
                        "plugin_op": plugin_data.get("op", plugin_name),
                    }
                )

    def match_subgraphs(self, model_path: str, output_config: Optional[str] = None):
        from polygraphy.tools.plugin.subtool.match import Match

        match_tool = Match()
        include_list = list(self.plugins.keys()) if self.plugins else []

        args = _ToolArgs(
            model_file=model_path,
            plugin_dir=self.plugin_dir,
            output=output_config,
            include=include_list,
            exclude=[],
        )

        match_tool.run_impl(args)
        config_path = output_config or os.path.join(
            os.path.dirname(model_path), "config.yaml"
        )

        if not os.path.exists(config_path):
            G_LOGGER.critical(f"Config file not found: {config_path}")

        self._load_config_from_yaml(config_path)

    def generate_config_for_subgraphs(
        self, selected_subgraphs: List[Dict[str, Any]], output_config_path: str
    ):
        # Group subgraphs by plugin name
        grouped_by_plugin = {}
        for subgraph in selected_subgraphs:
            plugin_name = subgraph["plugin_name"]
            grouped_by_plugin.setdefault(plugin_name, []).append(subgraph)

        # Build config data
        config_data = []
        for plugin_name, subgraphs in grouped_by_plugin.items():
            config_data.append(
                {
                    "name": plugin_name,
                    "op": subgraphs[0]["plugin_op"],
                    "instances": [
                        {
                            "inputs": subgraph["inputs"],
                            "outputs": subgraph["outputs"],
                            "attributes": subgraph.get("attributes", {}),
                        }
                        for subgraph in subgraphs
                    ],
                }
            )

        # Convert YAML to string and save
        from io import StringIO

        stream = StringIO()
        yaml.dump_all(config_data, stream, default_flow_style=False)
        config_content = stream.getvalue()
        util.save_file(
            config_content, output_config_path, mode="w", description="plugin config"
        )

        G_LOGGER.info(
            f"Generated config for {len(selected_subgraphs)} subgraph(s) to {output_config_path}"
        )

    def replace_subgraphs(
        self, model_path: str, config_path: str, output_path: Optional[str] = None
    ):
        from polygraphy.tools.plugin.subtool.replace import Replace

        replace_tool = Replace()

        args = _ToolArgs(
            model_file=model_path,
            plugin_dir=self.plugin_dir,
            output=output_path,
            config=config_path,
        )

        replace_tool.run_impl(args)
        output_file = output_path or os.path.join(
            os.path.dirname(model_path), "replaced.onnx"
        )

        if not os.path.exists(output_file):
            G_LOGGER.critical(f"Output file not found: {output_file}")

        G_LOGGER.info(f"Successfully replaced subgraphs in {output_file}")

    def get_all_subgraphs(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.plugin_subgraphs.copy()
