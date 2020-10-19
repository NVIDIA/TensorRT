## Quantization Basics

We use scale only symmetric quantization
```math
\mathbf{y_q}=rn(\frac{2^{k-1}-1}{\alpha}clip(\mathbf{y},-\alpha,\alpha))
```
where $`k`$ is number of bits to use for $`\mathbf{y_q}`$, $`\alpha`$ is absolute maximum quantized range of $`\mathbf{y}`$, *rn* is standard round to nearest, and *clip* is standard clip function which saturate values to $`[-\alpha, \alpha]`$. Note that $`\alpha`$ is not necessary a scalar, it can be any shape that can be broadcasted to $`\mathbf{y}`$. See https://arxiv.org/abs/2004.09602 for more detail. 

The key concept is $`\alpha`$ which defines quantization for given number of bits. It is named `amax` (maximum absolute value) throughout the entire code base.

"Fake" (emulated) quantization is usually used in quantization training. In "fake" mode, values are still represented in floating point, but only limited number of distinct values in a given range. For example, in 8bit, there are 255 values in $`[-\alpha, \alpha]`$ . Conceptually, "fake" mode is same as quantize then dequantize. It can be defined as
```math
\mathbf{y_{fq}}=\frac{\alpha}{2^{k-1}-1}rn(\frac{2^{k-1}-1}{\alpha}clip(\mathbf{y},-\alpha,\alpha))
```


## Install

```shell
git clone https://github.com/NVIDIA/TensorRT.git
cd tools/pytorch-quantization
python setup.py install
```

## Usage

#### Quantize an existing model

A model can be post training quantized by simply replacing `nn.ConvNd` and `nn.Linear` with `QuantConvNd` and `QuantLinear`. By default, it will be quantized to signed 8bit and compute amax in-flight for each batch. See a LeNet example below:

```python
import torch.nn as nn
import torch.nn.functional as F

from pytorch_quantization.nn import QuantConv2d, QuantLinear
from pytorch_quantization.tensor_quant import QuantDescriptor

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class QuantLeNet(nn.Module):
    def __init__(self):
        super(QuantLeNet, self).__init__()
        self.conv1 = QuantConv2d(1, 10, kernel_size=5)
        self.conv2 = QuantConv2d(10, 20, kernel_size=5)
        self.fc1 = QuantLinear(320, 50)
        self.fc2 = QuantLinear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

