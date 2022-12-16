# LayerNorm Plugin
 
**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)
 
## Description
 
The LayerNorm plugin implements the Layer Normalization operation described in the [Layer Normalization](https://arxiv.org/abs/1607.06450) paper.

$$
y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$

### Structure
 
This plugin has the plugin creator class `LayerNormPluginCreator` and the plugin class `LayerNormPlugin` which extends `IPluginV2DynamicExt`.
 
The LayerNorm plugin consumes the following inputs:
 
1. `input` with shape $(N_0, N_1, ..., N_{D-1})$: An input tensor.
2. `gamma` with shape $(N_{\text{axis}}, N_{\text{axis} + 1}, ..., N_{D-1})$: Scale ($\gamma$) to apply in the normalization operation.
3. `beta` with shape $(N_{\text{axis}}, N_{\text{axis} + 1}, ..., N_{D-1})$: Bias ($\beta$) to apply in the normalization operation.
 
The LayerNorm plugin produces the following output:
 
1. `output` with shape $(N_0, N_1, ..., N_{D-1})$: The normalized output.
 
## Parameters

The LayerNorm plugin has the following parameters:
 
| Type             | Parameter                       | Description
|------------------|---------------------------------|--------------------------------------------------------
|`float`             |`epsilon`                    | A value added in the denominator to the variance before calculating the standard deviation, for numerical stability. Default is 1e-5.
|`int`           |`axis`                    | The first dimension to normalize over. The allowed range is $[-D, D)$. A negative value means a dimension counted from the back. Default is -1.
 
 
## Additional resources
 
The following resources provide a deeper understanding of the LayerNorm plugin:
 
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
 
## License
 
For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.
 
## Changelog

 - January 2023: Parameter `axis` added to control the normalization shape. This is the first release of this `README.md` file.
 - November 2022: First release of the plugin. `README.md` file was absent. Normalization was always over the final `D-2` dimensions of the input. 

## Known issues

There are no known issues in this plugin.
