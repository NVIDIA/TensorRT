Quantizing Resnet50
===================

Create a quantized model
------------------------

Import the necessary python modules:

.. code:: python

   import torch
   import torch.utils.data
   from torch import nn

   from pytorch_quantization import nn as quant_nn
   from pytorch_quantization import calib
   from pytorch_quantization.tensor_quant import QuantDescriptor

   from torchvision import models

   sys.path.append("path to torchvision/references/classification/")
   from train import evaluate, train_one_epoch, load_data

Adding quantized modules
~~~~~~~~~~~~~~~~~~~~~~~~

The first step is to add quantizer modules to the neural network graph.
This package provides a number of quantized layer modules, which contain quantizers for inputs and weights.
e.g. ``quant_nn.QuantLinear``, which can be used in place of ``nn.Linear``.
These quantized layers can be substituted automatically, via monkey-patching, or by manually modifying the model definition.

Automatic layer substitution is done with ``quant_modules``. This should be called before model creation.

.. code:: python

    from pytorch_quantization import quant_modules
    quant_modules.initialize()

This will apply to all instances of each module.
If you do not want all modules to be quantized you should instead substitute the quantized modules manually.
Stand-alone quantizers can also be added to the model with ``quant_nn.TensorQuantizer``.

Post training quantization
--------------------------

For efficient inference, we want to select a fixed range for each quantizer.
Starting with a pre-trained model, the simplest way to do this is by calibration.


Calibration
~~~~~~~~~~~

We will use histogram based calibration for activations and the default max calibration for weights.

.. code:: python

   quant_desc_input = QuantDescriptor(calib_method='histogram')
   quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
   quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

   model = models.resnet50(pretrained=True)
   model.cuda()

To collect activation histograms we must feed sample data in to the model.
First, create ImageNet dataloaders as done in the training script.
Then, enable calibration in each quantizer and feed training data in to the model.
1024 samples (2 batches of 512) should be sufficient to estimate the distribution of activations.
Use training data for calibration so that validation also measures generalization of the selected ranges.

.. code:: python

    data_path = "PATH to imagenet"
    batch_size = 512

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    dataset, dataset_test, train_sampler, test_sampler = load_data(traindir, valdir, False, False)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size,
        sampler=test_sampler, num_workers=4, pin_memory=True)

.. code:: python

    def collect_stats(model, data_loader, num_batches):
        """Feed data to the network and collect statistic"""

        # Enable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
            model(image.cuda())
            if i >= num_batches:
                break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    def compute_amax(model, **kwargs):
        # Load calib result
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)
                print(F"{name:40}: {module}")
        model.cuda()

   # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model, data_loader, num_batches=2)
        compute_amax(model, method="percentile", percentile=99.99)

After calibration is done, quantizers will have ``amax`` set, which represents the absolute maximum input value representable in the quantized space.
By default, weight ranges are per channel while activation ranges are per tensor.
We can see the condensed amaxes by printing each ``TensorQuantizer`` module.

::

   conv1._input_quantizer                  : TensorQuantizer(8bit fake per-tensor amax=2.6400 calibrator=MaxCalibrator(track_amax=False) quant)
   conv1._weight_quantizer                 : TensorQuantizer(8bit fake axis=(0) amax=[0.0000, 0.7817](64) calibrator=MaxCalibrator(track_amax=False) quant)
   layer1.0.conv1._input_quantizer         : TensorQuantizer(8bit fake per-tensor amax=6.8645 calibrator=MaxCalibrator(track_amax=False) quant)
   layer1.0.conv1._weight_quantizer        : TensorQuantizer(8bit fake axis=(0) amax=[0.0000, 0.7266](64) calibrator=MaxCalibrator(track_amax=False) quant)
   ...

Evaluate the calibrated model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next we will evaluate the classification accuracy of our post training quantized model on the ImageNet validation set.

.. code:: python

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)

    # Save the model
    torch.save(model.state_dict(), "/tmp/quant_resnet50-calibrated.pth")

This should yield 76.1% top-1 accuracy, which is close to the pre-trained model accuracy of 76.2%.

Use different calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can try different calibrations without recollecting the histograms, and see which one gets the best accuracy.

.. code:: python

    with torch.no_grad():
        compute_amax(model, method="percentile", percentile=99.9)
        evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)

    with torch.no_grad():
        for method in ["mse", "entropy"]:
            print(F"{method} calibration")
            compute_amax(model, method=method)
            evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)


MSE and entropy should both get over 76%. 99.9% clips too many values for resnet50 and will get slightly lower accuracy.

Quantization Aware Training
---------------------

Optionally, we can fine-tune the calibrated model to improve accuracy further.

.. code:: python

   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
   lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

   # Training takes about one and half hour per epoch on a single V100
   train_one_epoch(model, criterion, optimizer, data_loader, "cuda", 0, 100)

   # Save the model
   torch.save(model.state_dict(), "/tmp/quant_resnet50-finetuned.pth")

After one epoch of fine-tuning, we can achieve over 76.4% top-1 accuracy.
Fine-tuning for more epochs with learning rate annealing can improve accuracy further.
For example, fine-tuning for 15 epochs with cosine annealing starting with a learning rate of 0.001 can get over 76.7%.
It should be noted that the same fine-tuning schedule will improve the accuracy of the unquantized model as well.

Further optimization
~~~~~~~~~~~~~~~~~~~~

For efficient inference on TensorRT, we need know more details about the runtime optimization.
TensorRT supports fusion of quantizing convolution and residual add.
The new fused operator has two inputs. Let us call them conv-input and residual-input.
Here the fused operator’s output precision must match the residual input precision.
When there is another quantizing node after the fused operator,
we can insert a pair of quantizing/dequantizing nodes between the residual-input and the Elementwise-Addition node,
so that quantizing node after the Convolution node is fused with the Convolution node, and the Convolution node is completely quantized with INT8 input and output.
We cannot use automatic monkey-patching to apply this optimization and we need to manually insert the quantizing/dequantizing nodes.

First create a copy of resnet.py from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py,
modify the constructor, add explicit bool flag ‘quantize’

.. code:: python

    def resnet50(pretrained: bool = False, progress: bool = True, quantize: bool = False, **kwargs: Any) -> ResNet:
        return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, quantize, **kwargs)
    def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], pretrained: bool, progress: bool,
                quantize: bool, **kwargs: Any) -> ResNet:
        model = ResNet(block, layers, quantize, **kwargs)
    class ResNet(nn.Module):
        def __init__(self,
                     block: Type[Union[BasicBlock, Bottleneck]],
                     layers: List[int],
                     quantize: bool = False,
                     num_classes: int = 1000,
                     zero_init_residual: bool = False,
                     groups: int = 1,
                     width_per_group: int = 64,
                     replace_stride_with_dilation: Optional[List[bool]] = None,
                     norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
            super(ResNet, self).__init__()
            self._quantize = quantize

When this ``self._quantize`` flag is set to ``True``, we need replace all the ``nn.Conv2d`` with ``quant_nn.QuantConv2d``.


.. code:: python

    def conv3x3(in_planes: int,
                out_planes: int,
                stride: int = 1,
                groups: int = 1,
                dilation: int = 1,
                quantize: bool = False) -> nn.Conv2d:
        """3x3 convolution with padding"""
        if quantize:
            return quant_nn.QuantConv2d(in_planes,
                                        out_planes,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=dilation,
                                        groups=groups,
                                        bias=False,
                                        dilation=dilation)
        else:
            return nn.Conv2d(in_planes,
                             out_planes,
                             kernel_size=3,
                             stride=stride,
                             padding=dilation,
                             groups=groups,
                             bias=False,
                             dilation=dilation)
      def conv1x1(in_planes: int, out_planes: int, stride: int = 1, quantize: bool = False) -> nn.Conv2d:
          """1x1 convolution"""
          if quantize:
              return quant_nn.QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
          else:
              return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


The residual conv add can be find both in both ``BasicBlock`` and ``Bottleneck``.
We need first declare quantization node in the ``__init__`` function.


.. code:: python

      def __init__(self,
                   inplanes: int,
                   planes: int,
                   stride: int = 1,
                   downsample: Optional[nn.Module] = None,
                   groups: int = 1,
                   base_width: int = 64,
                   dilation: int = 1,
                   norm_layer: Optional[Callable[..., nn.Module]] = None,
                   quantize: bool = False) -> None:
          # other code...
          self._quantize = quantize
          if self._quantize:
              self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)


Finally we need patch the ``forward`` function in both ``BasicBlock`` and ``Bottleneck``,
inserting extra quantization/dequantization nodes here.


.. code:: python

      def forward(self, x: Tensor) -> Tensor:
          # other code...
          if self._quantize:
              out += self.residual_quantizer(identity)
          else:
              out += identity
          out = self.relu(out)

          return out

The final resnet code with residual quantized can be found in https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/examples/torchvision/models/classification/resnet.py



