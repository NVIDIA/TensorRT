# Design

![](../images/trex_logo.png)

The `trex` package design follows the Model View Controller (MVC) design pattern:
* An `EnginePlan` instance represents a plan file and provides access to a Pandas dataframe which acts as the plan model. Each row in the dataframe represents one layer in the plan file, including its name, tactic, inputs and outputs and other attributes describing the layer.
    ```
    # Example: print plan layer names.
    plan = EnginePlan("my-engine.graph.json")
    df = plan.df
    print(df['Name'])
    ```

* An `EnginePlan` is constructed from three JSON files:
  * A plan-graph JSON file (mandatory) describes the engine plan in a JSON format.
  * A plan profiling file provides timing measurements in JSON format (optional; required only if you want to access timing information).
  * A metadata JSON file which describes the hardware and software enviroments on which the engine is profiled. This file is optional but provides valuable information about the engine.

    ```
    plan = EnginePlan(
        "my-engine.graph.json",
        "my-engine.profile.json",
        "my-engine.profile.metadata.json")
    ```

* A thin API provides access to views of the model. This is a convinience API on top of the Pandas dataframe. For example, `plan.get_layers_by_type` further preprocesses the dataframe for display. Other functions, like `group_count` provide dataframe groupding and reduction shortcuts.
    ```
    convs = plan.get_layers_by_type('Convolution')
    tactic_cnt = group_count(convs, 'tactic')
    ```

* There are APIs for plotting data (`plotting.py`), visualizing an engine graph (`graphing.py`), interactive notebooks (`interactive.py`, `notebook.py`) and easy-access reporting (`report_card.py`).

* The linting API is basic and in an early-preview status (`lint.py`).

# API Stability

`trex` is an experimental package and API stability is not guaranteed.  TensorRT's JSON format is controlled by TensorRT and is not likely to change much (except for extensions).
