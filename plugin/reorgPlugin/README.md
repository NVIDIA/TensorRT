# reorgPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The`reorgPlugin`  is specifically used for the reorg layer in the YOLOv2 model in TensorRT. It reorganizes the elements in the input tensor and generates an output tensor of a different shape. In YOLOv2, the output tensor from the reorg layer matches the shape of the output tensor from a downstream layer Conv20_1024 in the neural network. The two output tensors are then concatenated together as one single tensor.


### Structure

The `reorgPlugin` takes one input and generates one output. The tensor format must be in NCHW format.

The input is a tensor that has a shape of `[N, C, H, W]` where:
-   `N` is the batch size
-   `C` is the number of channels
-   `H` is the height of tensor
-   `W` is the width of the tensor
 
After a [unique one-to-one mapping](https://github.com/pjreddie/darknet/blob/8215a8864d4ad07e058acafd75b2c6ff6600b9e8/src/blas.c#L9), the output tensor of shape `[N, C x s x s, H / s, W / s]`, where s is the stride, is generated.

For example, if we have an input tensor of shape `[2, 4, 6, 6]`.
```
[[[[ 0 1 2 3 4 5]
[ 6 7 8 9 10 11]
[ 12 13 14 15 16 17]
[ 18 19 20 21 22 23]
[ 24 25 26 27 28 29]
[ 30 31 32 33 34 35]]

[[ 36 37 38 39 40 41]
[ 42 43 44 45 46 47]
[ 48 49 50 51 52 53]
[ 54 55 56 57 58 59]
[ 60 61 62 63 64 65]
[ 66 67 68 69 70 71]]

[[ 72 73 74 75 76 77]
[ 78 79 80 81 82 83]
[ 84 85 86 87 88 89]
[ 90 91 92 93 94 95]
[ 96 97 98 99 100 101]
[102 103 104 105 106 107]]

[[108 109 110 111 112 113]
[114 115 116 117 118 119]
[120 121 122 123 124 125]
[126 127 128 129 130 131]
[132 133 134 135 136 137]
[138 139 140 141 142 143]]]
  

[[[144 145 146 147 148 149]
[150 151 152 153 154 155]
[156 157 158 159 160 161]
[162 163 164 165 166 167]
[168 169 170 171 172 173]
[174 175 176 177 178 179]]

[[180 181 182 183 184 185]
[186 187 188 189 190 191]
[192 193 194 195 196 197]
[198 199 200 201 202 203]
[204 205 206 207 208 209]
[210 211 212 213 214 215]]

[[216 217 218 219 220 221]
[222 223 224 225 226 227]
[228 229 230 231 232 233]
[234 235 236 237 238 239]
[240 241 242 243 244 245]
[246 247 248 249 250 251]]

[[252 253 254 255 256 257]
[258 259 260 261 262 263]
[264 265 266 267 268 269]
[270 271 272 273 274 275]
[276 277 278 279 280 281]
[282 283 284 285 286 287]]]]
```
 
We set `stride = 2` and perform the reorganization, we will get the following output tensor of shape `[2, 16, 3, 3]`.
```
[[[[ 0 2 4]
[ 6 8 10]
[ 24 26 28]]

[[ 30 32 34]
[ 48 50 52]
[ 54 56 58]]

[[ 72 74 76]
[ 78 80 82]
[ 96 98 100]]

[[102 104 106]
[120 122 124]
[126 128 130]]

[[ 1 3 5]
[ 7 9 11]
[ 25 27 29]]

[[ 31 33 35]
[ 49 51 53]
[ 55 57 59]]

[[ 73 75 77]
[ 79 81 83]
[ 97 99 101]]

[[103 105 107]
[121 123 125]
[127 129 131]]

[[ 12 14 16]
[ 18 20 22]
[ 36 38 40]]

[[ 42 44 46]
[ 60 62 64]
[ 66 68 70]]

[[ 84 86 88]
[ 90 92 94]
[108 110 112]]

[[114 116 118]
[132 134 136]
[138 140 142]]

[[ 13 15 17]
[ 19 21 23]
[ 37 39 41]]

[[ 43 45 47]
[ 61 63 65]
[ 67 69 71]]

[[ 85 87 89]
[ 91 93 95]
[109 111 113]]

[[115 117 119]
[133 135 137]
[139 141 143]]]
  

[[[144 146 148]
[150 152 154]
[168 170 172]]

[[174 176 178]
[192 194 196]
[198 200 202]]

[[216 218 220]
[222 224 226]
[240 242 244]]

[[246 248 250]
[264 266 268]
[270 272 274]]

[[145 147 149]
[151 153 155]
[169 171 173]]

[[175 177 179]
[193 195 197]
[199 201 203]]

[[217 219 221]
[223 225 227]
[241 243 245]]

[[247 249 251]
[265 267 269]
[271 273 275]]

[[156 158 160]
[162 164 166]
[180 182 184]]

[[186 188 190]
[204 206 208]
[210 212 214]]

[[228 230 232]
[234 236 238]
[252 254 256]]

[[258 260 262]
[276 278 280]
[282 284 286]]

[[157 159 161]
[163 165 167]
[181 183 185]]

[[187 189 191]
[205 207 209]
[211 213 215]]

[[229 231 233]
[235 237 239]
[253 255 257]]

[[259 261 263]
[277 279 281]
[283 285 287]]]]
```

## Parameters

The `reorgPlugin` has plugin creator class `ReorgPluginCreator` and plugin class `Reorg`.

The following parameters were used to create the `Reorg` instance.

| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`int`     |`stride`                  |Dimension reduction factor for the input tensor. The stride has to be divisible by the height and the width of the input tensor.


## Additional resources

The following resources provide a deeper understanding of the `regionPlugin` plugin:

- [YOLOv2 paper](https://arxiv.org/abs/1612.08242) 
- [Reorg layer in YOLOv2](https://github.com/pjreddie/darknet/blob/8215a8864d4ad07e058acafd75b2c6ff6600b9e8/src/blas.c#L9)
- [YOLOv2 architecture](https://ethereon.github.io/netscope/#/gist/d08a41711e48cf111e330827b1279c31)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

May 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.