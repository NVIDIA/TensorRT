# **Introduction to Quantization**

## What is Quantization?
[Quantization](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8) is the process of converting continuous values to discrete set of values using linear/non-linear scaling techniques.

## Why Quantization?

* High precision is necessary during training for fine-grained weight updates.

* High precision is not usually necessary during inference and may hinder the deployment of AI models in real-time and/or in resource-limited devices.

* INT8 is computationally less expensive and has lower memory footprint.

* INT8 precision results in faster inference with similar performance.

## Quantization Basics
See [whitepaper](https://arxiv.org/abs/2004.09602) for more detailed explanations.

Let [&beta;, &alpha;] be the range of representable real values chosen for quantization and *`b`* be the bit-width of the signed integer representation.  
The goal of uniform quantization is to map real values in the range [&beta; , &alpha;] to lie within [-2<sup>b-1</sup>, 2<sup>b-1</sup> - 1]. The real values that lie outside this range are clipped to the nearest bound.  

*Affine Quantization*

Considering 8 bit quantization (*b=8*), a real value within range [&beta;, &alpha;] is quantized to lie within the quantized range `[-128, 127]` (see [source](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Layers.html#iquantizelayer)):

x<sub>q</sub>=clamp(round(x/scale)+zeroPt)

where,
scale = (&alpha; - &beta;) / (2<sup>b</sup>-1)  

zeroPt = -round(&beta; * scale) - 2<sup>b-1</sup>  

`round` is a function that rounds a value to the nearest integer. The quantized value is then clamped between -128 to 127.

*Affine DeQuantization*

DeQuantization is the reverse process of quantization (see [source](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Layers.html#idequantizelayer)):

x=(x<sub>q</sub>−zeroPt)∗scale

## Quantization in TensorRT  

[TensorRT(TM)](https://developer.nvidia.com/tensorrt-getting-started) only supports symmetric uniform quantization, meaning that `zeroPt=0` (i.e. the quantized value of 0.0 is always 0).

Considering 8 bit quantization (*`b=8`*), a real value within range [`min_float`, `max_float`] is quantized to lie within the quantized range `[-127, 127]`, opting not to use `-128` in favor of symmetry. It is important to note that we loose 1 value in symmetric quantization representation, however, loosing 1 out of 256 representable value for 8 bit quantization is insignificant.

*Quantization*

The mathematical representation for symmetric quantization (`zeroPt=0`) is:

x<sub>q</sub>=clamp(round(x/scale))

Since TensorRT supports only symmetric range, the scale is calculated using the max absolute value: `max(abs(min_float), abs(max_float))`. 

Let &alpha; = `max(abs(min_float), abs(max_float))`,

scale = &alpha;/(2<sup>b-1</sup>-1)

Rounding [type](https://en.wikipedia.org/wiki/Rounding#Round_half_to_even) is rounding-to-nearest ties-to-even.
The quantized value is then clamped between `-127` and `127`.

*DeQuantization*
Symmetric dequantization is the reverse process of symmetric quantization:

x=(x<sub>q</sub>)∗scale


## Intutions

### Quantization Scale

Scaling factor divides a given range of real values into a number of partitions.

Lets understand intution behind scaling factor formula by taking 3 bit quantization as an example. 

*Asymmetric Quantization*

Real values range: [&beta;, &alpha;]

Quantized values range: [-2<sup>3-1</sup>, 2<sup>3-1</sup>-1]  
i.e. [-4, -3, -2, -1, 0, 1, 2, 3]

As expected there are 8 quantized (2<sup>3</sup>) values for 3 bit quantization.

Scale divides range into partitions. There are 7 (2<sup>3</sup>-1) partitions for 3 bit quantization.
Thus,  
scale = (&alpha; - &beta;) / (2<sup>3</sup>-1)

*Symmetric Quantization*

Symmetric quantization brings in two changes

1. Real values are not free now but are restricted. i.e [-&alpha;, &alpha;]  
where &alpha; = `max(abs(min_float), abs(max_float))`
2. One value from quantization range is dropped in favor of symmetry leading to a new range [-3, -2, -1, 0, 1, 2, 3].  

There are now 6 (2<sup>3</sup>-2) partitions (unlike 7 for asymmetric quantization).  

Scale divides range into partitions.

scale = 2*&alpha; /(2<sup>3</sup> - 2) = &alpha;/(2<sup>3-1</sup>-1)  

Similar intution holds true for `b` bit quantization.

### Quantization Zero Point

The constant `zeroPt` is of the same type as quantized values x<sub>q</sub>, and is in fact the quantized value x<sub>q</sub> corresponding to the real value 0. This allows us to auto-matically meet the requirement that the real value r = 0 be exactly representable by a quantized value. The motivation for this requirement is that efficient implementation of neural network operators often requires zero-padding of arrays around boundaries.

If we have values with negative data, then the zero point can offset the range. So if our zero point was 128, then unscaled negative values -127 to -1 would be represented by 1 to 127, and positive values 0 to 127 would be represented by 128 to 255.