# Using Custom Input Data

For any tools that use inference input data, such as `run` or `convert`, Polygraphy
provides 2 ways to supply custom input data:

1. `--load-inputs`/`--load-input-data`, which takes a path to a JSON file containing a
    `List[Dict[str, np.ndarray]]`.
    The JSON file should be created by using Polygraphy's JSON utilities, like `save_json`,
    in the `polygraphy.json` submodule.

    *NOTE: This will cause Polygraphy to load the entire object into memory and so may be*
        *impractical or impossible if the data is very large.*

2. `--data-loader-script`, which takes a path to a Python script that defines a `load_data` function
    that returns a data loader. The data loader can be any iterable or generator that yields
    `Dict[str, np.ndarray]`. By using a generator, we can avoid loading all the data
    at once, and instead limit it to just a single input at a time.

    *TIP: If you have an existing script that already defines such a function, you do **not** need to create*
        *a separate script just for the sake of `--data-loader-script`. You can simply use the existing script*
        *and specify the name of the function if it's not `load_data`*


## Further Reading

- See [`run` example 05](../examples/cli/run/05_comparing_with_custom_input_data/)
    for examples of both approaches highlighted above.
