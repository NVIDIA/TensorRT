# Matching and replacing a subgraph with a plugin in an onnx model

## Introduction

The `plugin` tool offers subtools to find and replace subgraphs in an onnx model.

Subgraph substition is a three-step process:

1. Find matching subgraphs based on the plugin's graph pattern (pattern.py) and list the potential substitutions in a user-editable intermediate file (config.yaml)
2. Review and edit (if necessary) the list of potential substitutions (config.yaml)
3. Replace subgraphs with plugins based on the list of potential substitutions (config.yaml)

`original.onnx` -------> `match` -------> `config.yaml` -------> `replace` -------> `replaced.onnx`
`plugins` ----------------^ `usr input`---^ `plugins`--------^

## Details

### Match

Finding matchings subgraphs in a model is done based on a graph pattern description (`pattern.py`) provided by the plugins.
The graph pattern description (`pattern.py`) contains information about the topology and additional constraints for the graph nodes, and a way to calculate the plugin's attributes based on the matching subgraph.
Only plugins which provide a graph pattern description (pattern.py) are considered for matching.

The result of the matching is stored in an intermediate file called `config.yaml`.
The user should review and edit this file, as it serves as a TODO list for the replacement step. For example, if there are 2 matching subgraphs, but only one should be substituted, the result can be removed from the file.

As a preview/dry-run step, the `plugin list` subtool can show the list of potential substitutions without generating an intermediate file.

### Replace

Replacement of subgraphs with plugins uses the `config.yaml` file generated in the matching stage. Any matching subgraph listed in this file is going to be removed and replaced with a single node representing the plugin. The original file is kept, and a new file is saved where the replacements are done. This file by default is called `replaced.onnx`.

### Compare

The original and the replaced model can be compared to check if they behave the same way before and after plugin substitution:
`polygraphy run original.onnx --trt --save-outputs model_output.json`
`polygraphy run replaced.onnx --trt --load-outputs model_output.json`

## Running The Example

1. Find and save matches of toyPlugin in the example network:

   ```bash
   polygraphy plugin match toy_subgraph.onnx \
       --plugin-dir ./plugins -o config.yaml
   ```

   <!-- Polygraphy Test: Ignore Start -->

   This will display something like:

   ```
   checking toyPlugin in model
   [I] Start a subgraph matching...
   [I] 	Checking node: n1 against pattern node: Anode.
   [I] 	No match because: Op did not match. Node op was: O but pattern op was: A.
   [I] Start a subgraph matching...
   [I] Found a matched subgraph!
   [I] Start a subgraph matching...
   ```

   The resulting config.yaml will look like:

   ```
   name: toyPlugin
   instances:
   - inputs:
   - i1
   - i1
   outputs:
   - o1
   - o2
   attributes:
       x: 1
   ```

   <!-- Polygraphy Test: Ignore End -->

2. **[Optional]** List matches of toyPlugin in the example network, without saving config.yaml:

   ```bash
   polygraphy plugin list toy_subgraph.onnx \
       --plugin-dir ./plugins
   ```

   <!-- Polygraphy Test: Ignore Start -->

   This will display something like:

   ```
   checking toyPlugin in model
   [I] Start a subgraph matching...
   [I] 	Checking node: n1 against pattern node: Anode.
   [I] 	No match because: Op did not match. Node op was: O but pattern op was: A.
   [I] Start a subgraph matching...
   ...
   [I] Found a matched subgraph!
   [I] Start a subgraph matching...
   [I] 	Checking node: n6 against pattern node: Anode.
   [I] 	No match because: Op did not match. Node op was: E but pattern op was: A.
   the following plugins would be used:
   {'toyPlugin': 1}
   ```

   There will be no resulting config.yaml, as this command is only for printing the number of matches per plugin
   <!-- Polygraphy Test: Ignore End -->

The `plugin replace` subtool replaces subgraphs in an onnx model with plugins

3. Replace parts of the example network with toyPlugin:

   ```bash
   polygraphy plugin replace toy_subgraph.onnx \
       --plugin-dir ./plugins --config config.yaml -o replaced.onnx
   ```

   <!-- Polygraphy Test: Ignore Start -->

   This will display something like:

   ```
   [I] Loading model: toy_subgraph.onnx
   ```

   The result file is replaced.onnx, where a subgraph in the example network is replaced by toyPlugin
   <!-- Polygraphy Test: Ignore End -->
