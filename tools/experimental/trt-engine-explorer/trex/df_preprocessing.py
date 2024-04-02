#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This file contains raw JSON dataframes preprocessing functions.
"""


import pandas as pd
from .activations import *


layer_attributes = {
    'Convolution': {
        'subtype': 'subtype',
        'Groups': 'attr.groups',
        'OutMaps': 'attr.out_maps',
        'Stride': 'attr.stride',
        'Kernel': 'attr.kernel',
        'HasBias': 'attr.has_bias',
        'Activation': 'attr.activation',
        'HasReLU': 'attr.has_relu',
        'PaddingMode': None, # 'attr.padding_mode',
        'PrePadding': None, # 'attr.pre_padding',
        'PostPadding': None, # 'attr.post_padding',
        'Dilation': 'attr.dilation',
        'AllowSparse': None,
        'Weights': 'Weights',
        'Bias': None,
        'Dimensions': None,
        'ConvolutionTacticIndex': None,
        'Operations': 'attr.operations',
        'NbOperations': 'attr.n_operations',
    },

    'PointWiseV2': {
        'Operations': 'attr.operations',
        'NbOperations': 'attr.n_operations',
        'NbInputArgs': 'attr.n_input_args',
        'InputArgs': None,
        'NbOutputVars': None,
        'OutputVars': None,
        'NbLiterals': None,
        'Literals': None,
        'NbParams':  None,
        'Params': None,
    },

    'PointWise': {
        'Instructions': 'attr.operations',
        'NbInstructions': 'attr.n_operations',
        'NbInputs': 'attr.n_input_args',
        'InputArgs': None,
        'NbOutputVars': None,
        'OutputVars': None,
        'NbLiterals': None,
        'Literals': None,
        'NbParams':  None,
        'Params': None,
    },
    'Reformat': {
        'Origin': 'attr.origin',
    },

    'Pooling': {
        'PoolingType': 'attr.pooling_type',
        'WindowSize': 'attr.window_size',
        'AverageCountExcludesPadding': None,
        'PaddingMode': 'attr.padding_mode',
        'Stride': 'attr.stride',
        'PrePadding': None,
        'PostPadding': None,
        'BlendFactor': None,
    },

    'Scale': {
        'Mode': 'attr.mode',
        'Shift': 'attr.shift',
        'Scale': 'attr.scale',
        'Power': 'attr.power',
        'Activation': 'attr.activation',
        'ChannelAxis': 'attr.ch_axis',
    },

    'Shuffle': {
        'FirstTranspose': 'attr.first_transpose',
        'SecondTranspose': 'attr.second_transpose',
        'Reshape': 'attr.reshape',
        'ZeroIsPlaceholder': None,
        'ParameterSubType': None,
    },

    'Resize': {
        'ResizeMode': 'attr.mode',
        'ResizeScales': 'attr.scales',
        'NNRounding': 'attr.scale',
        'Start': None,
        'CoordTransform': None,
        'ResizeSelector': None,
    },

    'Slice': {
        'Start': 'attr.start',
        'Stride': 'attr.stride',
        'Size': 'attr.size',
        'Mode': 'attr.mode',
        'negativeInfinityPadding': None,
    },

    'Common': {
        'ParameterType': None,
        'weights': None,
        'dimensions': None,
        'TacticValue': None,
    },
}


def __fix_type(df: pd.DataFrame):
    df.rename(columns={'LayerType': 'subtype'}, inplace=True)
    try:
        df['type'] = df.ParameterType.fillna(value=df.subtype)
        df.drop(['ParameterType'], axis=1, inplace=True)
    except AttributeError:
        pass


def __fix_tactic(df: pd.DataFrame):
    df.rename(columns={'TacticName': 'tactic'}, inplace=True)
    try:
        df['tactic'] = df.tactic.fillna(value='TensorRT')
    except AttributeError:
        df['tactic'] = 'TensorRT'


def __fix_columns_types(df: pd.DataFrame):
    int_cols = [
        'Groups', 'OutMaps', 'HasBias', 'HasReLU', 'AllowSparse',
        'NbInputArgs', 'NbOutputVars', 'NbParams', 'NbLiterals', ]
    for col in int_cols:
        try:
            df[col] = df[col].fillna(value=0)
            df[col] = df[col].astype('int32')
        except KeyError:
            pass
    df.fillna(0, inplace=True)


def __fix_output_precision(df: pd.DataFrame):
    fixed_outputs = []
    for outputs in df['Outputs']:
        try:
            fixed_outputs.append(Activation(outputs[0]).precision)
        except IndexError:
            # Some layers may have empty outputs.
            fixed_outputs.append('')
    df['output_precision'] = fixed_outputs


def fix_df(df: pd.DataFrame) -> pd.DataFrame:
    """One-time preprocessing of the DF.

    Performed only on DF construction.
    """
    __fix_type(df)
    __fix_tactic(df)
    __fix_columns_types(df)
    __fix_output_precision(df)
    return df


def clean_io(df: pd.DataFrame):
    for index, layer in df.iterrows():
        inputs, outputs = create_activations(layer)
        if len(inputs)  > 0:
            inp_str = ", ".join([inp.format for inp in inputs])
            df.loc[index, 'Inputs'] = inp_str
        df.loc[index, 'Outputs'] = outputs[0].format


def filter_by_layer(df: pd.DataFrame, layer_type: str) -> pd.DataFrame:
    copy_cols = ['Name', 'type', 'precision', 'tactic',
                 'latency.pct_time', 'latency.avg_time',
                 'total_io_size_bytes', 'total_footprint_bytes',
                 'Inputs', 'Outputs', 'subtype']
    try:
        attrs = layer_attributes[layer_type]
        copy_cols += [k for k, v in attrs.items() if v is not None]
        # Pointwise and Pointwise V2 layers have the same layer type.
        if layer_type == 'PointWise':
            attrs = layer_attributes['PointWiseV2']
            copy_cols += [k for k, v in attrs.items() if v is not None]
    except KeyError:
        pass

    layers = df.query(f"type == \"{layer_type}\"").copy()
    if len(layers) == 0:
        return layers

    copy_cols = list(set(copy_cols) & set(layers.columns))
    layers = layers[copy_cols]

    if layer_type == 'Convolution':
        layers.rename(columns=layer_attributes[layer_type], inplace=True)
        layers['attr.kernel'] = tuple(layers['attr.kernel'])
        annotate_convolutions(layers)
    if layer_type == 'PointWise':
        # The original JSON file handle PointWise and PointWise V2 as two subtypes.
        pw = layers[layers['subtype'] == 'PointWise'].copy()
        pw.rename(columns=layer_attributes['PointWise'], inplace=True)
        pw_v2 = layers[layers['subtype'] == 'PointWiseV2'].copy()
        pw_v2.rename(columns=layer_attributes['PointWiseV2'], inplace=True)
        layers = pd.concat((pw, pw_v2))
    else:
        try:
            layers.rename(columns=layer_attributes[layer_type], inplace=True)
        except:
            pass
    return layers


def change_col_order(df: pd.DataFrame) -> pd.DataFrame:
    """Change the dataframe columns-order (place common fields earlier)"""
    cols = df.columns.to_list()
    common_cols = list(('Name', 'type', 'Inputs', 'Outputs', 'latency.avg_time',
        'latency.pct_time', 'total_footprint_bytes', 'tactic'))
    common_cols = [col for col in common_cols if col in cols]
    cols = common_cols + [col for col in cols if col not in common_cols]
    df = df[cols]
    return df


def drop_columns(df: pd.DataFrame, columns: list):
    for col in columns:
        try:
            df.drop([col], axis=1, inplace=True)
        except KeyError:
            pass


def clean_df(df: pd.DataFrame, inplace=True) -> pd.DataFrame:
    clean_io(df)
    columns = set([col for col_list in layer_attributes.keys() for col in col_list])
    drop_columns(df, columns)
    df.fillna(0, inplace=inplace)
    return df


def clean_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the dataframe for display"""
    df = clean_df(df.copy(), inplace=True)
    df = change_col_order(df)
    drop_columns(df,
        columns=['subtype', 'TacticValue', 'precision', 'total_io_size_bytes'])
    return df


def annotate_convolutions(convs: pd.DataFrame):
    """Convolutions as implicit GEMM"""
    for index, conv in convs.iterrows():
        inputs, outputs = create_activations(conv)
        assert len(inputs)  > 0
        N, C, H, W = inputs[0].shape
        # K: number of channels; P: Height; Q: Width
        _, K, P, Q = outputs[0].shape
        R, S = convs.loc[index, 'attr.kernel']
        G = convs.loc[index, 'attr.groups']
        weights_vol = (K * C * R * S) / G
        input_vol = N * C * H * W
        output_vol = N * K * P * Q
        input_bytes = input_vol * inputs[0].data_size
        output_bytes = output_vol * outputs[0].data_size
        weights_bytes = weights_vol * inputs[0].data_size
        nb_bytes = input_bytes + weights_bytes + output_bytes
        nb_macs = N * K * P * Q * C * R * S / G
        convs.loc[index, 'attr.macs'] = nb_macs
        # Arithmetic intensity: ops/bytes
        convs.loc[index, 'attr.arithmetic_intensity'] = nb_macs / nb_bytes
        latency = convs.loc[index, 'latency.avg_time']
        if latency > 0:
            convs.loc[index, 'attr.compute_efficiency'] = nb_macs / latency
            convs.loc[index, 'attr.memory_efficiency'] = nb_bytes / latency
        else:
            convs.loc[index, 'attr.compute_efficiency'] = 0
            convs.loc[index, 'attr.memory_efficiency'] = 0
        # Conversion to matrices (M, K) * (K, N)
        M = N * P * Q
        N = K
        K = C * R * S
        convs.loc[index, 'attr.M'] = M
        convs.loc[index, 'attr.N'] = N
        convs.loc[index, 'attr.K'] = K

    convs['attr.macs'] = convs['attr.macs'].astype('int64')
    convs['attr.M'] = convs['attr.M'].astype('int64')
    convs['attr.N'] = convs['attr.N'].astype('int64')
    convs['attr.K'] = convs['attr.K'].astype('int64')
