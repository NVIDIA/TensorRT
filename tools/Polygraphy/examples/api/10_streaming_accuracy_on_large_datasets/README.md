# Streaming Accuracy Comparison On Large Datasets


## Introduction

For a general overview of comparing framework outputs with the Polygraphy API, see the example on
[Comparing Frameworks](../01_comparing_frameworks/).

`Comparator.run()` followed by `Comparator.compare_accuracy()` holds every iteration's outputs in
memory, which becomes impractical on large datasets. The streaming API avoids this:

- `Comparator.run(runners, data_loader=..., streaming=True)` is a generator that runs inference one
  input at a time, activating all runners simultaneously and discarding the outputs after each
  iteration, so memory usage stays roughly constant regardless of dataset size.
- `Comparator.compare_accuracy(streams, ...)` accepts a *list* of such streams instead of a
  materialized run, comparing each iteration as it is produced and accumulating only the (tiny)
  accuracy metrics. For a single live run, pass `[Comparator.run(runners, streaming=True)]`. It
  returns an `AccuracyResults` whose `bool(...)` is True only if every result passed.

In this example, we use a lazy [data loader](./data_loader.py) (which yields inputs one at a time
without materializing the whole dataset) to:

1. Stream two runs of an identity model over a large dataset and compare them as we go.
2. Stream a fresh run and compare it against reference ("golden") outputs saved on disk -- still
   one iteration at a time.


## Running The Example

1. Install prerequisites
    * Install dependencies with `python3 -m pip install -r requirements.txt`

2. Generate the reference ("golden") outputs. In a real workflow these would already exist (from a
    trusted backend, a previous release, or a reference implementation), so this step just stands in
    for that:
    ```bash
    python3 generate_golden.py
    ```

3. Run the example
    ```bash
    python3 example.py
    ```

4. **[Optional]** Inspect the saved golden outputs:

    <!-- Polygraphy Test: Ignore Start -->
    ```bash
    polygraphy inspect data golden
    ```
    <!-- Polygraphy Test: Ignore End -->


## Understanding The Code

`generate_golden.py` saves a run while streaming: an *extensionless* `save_outputs_path` (here,
`golden`) is a *directory* holding one JSON file per iteration (it must be empty or not yet exist),
while a path *with* a file extension (e.g. `golden.json`) instead holds the whole run in a single
file. `example.py` reads it back one iteration at a time with `RunResults.load_streaming(<dir>)`.

To compare a live run against a saved one, pass both streams to `compare_accuracy`:

<!-- Polygraphy Test: Ignore Start -->
```py
Comparator.compare_accuracy(
    [
        Comparator.run(runners, data_loader=load_data(), streaming=True),
        RunResults.load_streaming("golden"),
    ],
)
```
<!-- Polygraphy Test: Ignore End -->

The streams are merged per iteration (each stream's runners appended in order), so the live run is
compared against the golden run one iteration at a time.


## Further Reading

For the CLI equivalent of this workflow, see
[`run` example 09](../../cli/run/09_streaming_accuracy_on_large_datasets/). For non-streaming
output comparison and metric-based comparison functions (L2, PSNR, SNR, ...), see
[API example 01](../01_comparing_frameworks/). For working with saved run results manually, see
[API example 08](../08_working_with_run_results_and_saved_inputs_manually/).
