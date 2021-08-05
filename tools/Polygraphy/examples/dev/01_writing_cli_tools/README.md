# Writing Custom Command-Line Tools


## Introduction

Polygraphy includes various helper utilities to make it easier to write
new command-line tools from scratch.

In this example, we'll write a brand new tool called `gen-data` that will generate random data
using Polygraphy's default data loader, and then write it to an output file. The user will
be able to specify the number of values to generate as well as the output path.

To do this, we'll create a child class of `Tool` and use the `DataLoaderArgs` argument
group provided by Polygraphy.


## Running The Example

1. You can run the example tool from this directory. For example:

    ```bash
    ./gen-data -o data.json --num-values 25
    ```

2. We can even inspect the generated data with `inspect data`:

    ```bash
    polygraphy inspect data data.json -s
    ```

To see the other command-line options available in the example tool,
run:
```bash
./gen-data -h
```
