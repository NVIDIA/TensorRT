# Contributing guidelines

## Style

#### C++ coding style

Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html), with max-line-length extended to 120.

Run `cpplint` before committing code.

#### Python coding style

Follow [Google Python Style Guide,](https://google.github.io/styleguide/pyguide.html) with max-line-length extended to 120. Exceptions are allowed if it feels more natural to follow Pytorch style. For example, Pytorch allows import relative path, also class name.

Run `pylint` before committing code. It doesn't mean every issue has to be corrected nor check has to be manually disabled. Just make sure you are aware of the remaining issues and you are comfort with all of them. But, don't leave lint error there. Disable it explicitly if it is not a real error.

Install `pylint`

```bash
pip install pylint
```

To check a file with `pylint`:

```bash
pylint --rcfile=.pylintrc myfile.py
```

#### Yapf

[yapf](https://github.com/google/yapf/) is an auto format tool owned by Google (not a Google product). To save the time of arguing code style during code review, use yapf to format the code is a good option. Note that it doesn't reformat comment.

Install `yapf`

```bash
pip install yapf
```

Format code with yapf

```bash
yapf myfile.py --style .style.yapf
```

There are Sublime and Vim plugins.

## Test

Use [googletest](https://github.com/google/googletest) for c++ code.

Use [pytest](https://docs.pytest.org/en/latest/) for python code.

To run all the tests:

```shell
pytest --verbose
```

To run a particular test file

```shell
pytest --verbose mytestfile.py
```



## Conventions

Quantization is a very overloaded word, many things related to it can create a lot of confusions. Let's try to avoid confusions as much as possible by following existing conventions. Generally, if there is a similar Tensorflow or numpy function, follow its convention. Though Tensorflow uses `quantized`, `quantization` and `quant`, let's stick with the shortest one only.

### Naming

##### Function and class name

When developing quantized version of a function or module, add`Quant` to class name, add `quant_` to function name, e.g.

```python
class Linear(...)
class QuantLinear(...)

def linear(...)
def quant_linear(...)
```

##### Variable name

Add prefix `quant_mode_`, `num_bits_`  etc. to name of tensors will be quantized, e.g.

```python
def matmul(a, b)
def quant_matmul(a, b)
```

Don't use prefix/suffix `weight` or `act` if tensor being quantized doesn't have them explicitly in name. From function's perspective, it takes tensors, not necessarily weight and activation tensors. e.g. `a` and `b` of `matmul` can both be either weight or activation.

##### Quantization mode

There only convention here we can adopt is `per_channel`. Other things, like there is no convention to follow of per row/column scale of matrix multiply. And though we usually absolute max value based scaling factor, there are other ways to decide it, like KL-divergence. 

Our API design is flexible enough to support any granularity of quantization. The main concept is `axis`.

```python
# axis=None means per tensor
# For 2d convolution weight with layout KCRS, axis=(1, 2, 3) means perchannel quantization
# more example below
QUANT_DESC_8BIT_PER_TENSOR = QuantDescriptor(num_bits=8)
QUANT_DESC_UNSIGNED_8BIT_PER_TENSOR = QuantDescriptor(num_bits=8, unsigned=True)
QUANT_DESC_8BIT_CONV1D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV3D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE1D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE3D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))

```

### Misc

