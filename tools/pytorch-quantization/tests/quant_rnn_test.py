#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""tests of QuantRNN module.
"""
import pytest

import torch
from torch import nn

import numpy as np

from pytorch_quantization.nn.modules import quant_rnn
from pytorch_quantization import tensor_quant

from tests.fixtures import verbose

from . import utils

# make everything run on the GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# change default type to double if utils.compare flags a small error, may just be floating point rounding error
# torch.set_default_tensor_type('torch.cuda.DoubleTensor')

np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

# pylint: disable=no-self-use, missing-docstring, redefined-builtin, bad-continuation

# global state for saving/loading test vectors
SAVE_VECTORS = 0
VECTOR_FILE = 'tests/quant_rnn_test_vectors.pt'
if SAVE_VECTORS:
    TEST_VECTORS = dict()
else:
    TEST_VECTORS = torch.load(VECTOR_FILE)


class TestQuantLSTMCell():
    """
    tests for quant_rnn.QuantLSTMCell
    default parameters in QuantLSTMCell:
        bias=True,
        num_bits_weight=8, quant_mode_weight='per_channel',
        num_bits_input=8, quant_mode_input='per_tensor'

    Tests of real quantization mode (nonfake) are disabled as it is not fully supported yet.
    """

    def test_basic_forward(self, verbose):
        """Do a forward pass on the cell module and see if anything catches fire."""
        batch = 7
        input_size = 11
        hidden_size = 9

        quant_desc_input = tensor_quant.QuantDescriptor(num_bits=8)
        quant_desc_weight = tensor_quant.QuantDescriptor(num_bits=8, axis=(1,))
        quant_rnn_object = quant_rnn.QuantLSTMCell(input_size, hidden_size, bias=False,
                quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)
        quant_rnn_object._input_quantizer.disable()
        quant_rnn_object._weight_quantizer.disable()

        input = torch.randn(batch, input_size)
        hidden = torch.randn(batch, hidden_size)
        cell = torch.randn(batch, hidden_size)

        quant_rnn_object(input, hx=(hidden, cell))

    def test_no_quant_input_hidden(self, verbose):
        """QuantLSTM with quantization disabled vs. pytorch LSTM for input and hidden inputs."""
        batch = 17
        input_size = 13
        hidden_size = 7

        quant_rnn_object = quant_rnn.QuantLSTMCell(input_size, hidden_size, bias=False)
        quant_rnn_object._input_quantizer.disable()
        quant_rnn_object._weight_quantizer.disable()
        ref_rnn_object = nn.LSTMCell(input_size, hidden_size, bias=False)

        # copy weights from one rnn to the other
        ref_rnn_object.load_state_dict(quant_rnn_object.state_dict())

        input = torch.randn(batch, input_size)
        hidden = torch.randn(batch, hidden_size)
        cell = torch.randn(batch, hidden_size)

        quant_hout, quant_cout = quant_rnn_object(input, hx=(hidden, cell))
        ref_hout, ref_cout = ref_rnn_object(input, hx=(hidden, cell))

        utils.compare(quant_hout, ref_hout)
        utils.compare(quant_cout, ref_cout)

    def test_no_quant_input_hidden_bias(self, verbose):
        """QuantLSTMCell with quantization disabled vs. pytorch LSTMCell for input, hidden inputs and bias."""
        batch = 19
        input_size = 11
        hidden_size = 3

        quant_rnn_object = quant_rnn.QuantLSTMCell(input_size, hidden_size, bias=True)
        quant_rnn_object._input_quantizer.disable()
        quant_rnn_object._weight_quantizer.disable()
        ref_rnn_object = nn.LSTMCell(input_size, hidden_size, bias=True)

        # copy weights from one rnn to the other
        ref_rnn_object.load_state_dict(quant_rnn_object.state_dict())

        input = torch.randn(batch, input_size)
        hidden = torch.randn(batch, hidden_size)
        cell = torch.randn(batch, hidden_size)

        quant_hout, quant_cout = quant_rnn_object(input, hx=(hidden, cell))
        ref_hout, ref_cout = ref_rnn_object(input, hx=(hidden, cell))

        utils.compare(quant_hout, ref_hout)
        utils.compare(quant_cout, ref_cout)

    def test_against_unquantized(self, verbose):
        """Quantization should introduce bounded error utils.compare to pytorch implementation."""
        batch = 9
        input_size = 13
        hidden_size = 7

        quant_desc_input = tensor_quant.QuantDescriptor(num_bits=16)
        quant_desc_weight = tensor_quant.QuantDescriptor(num_bits=16, axis=(1,))
        quant_rnn_object = quant_rnn.QuantLSTMCell(input_size, hidden_size, bias=False,
                quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)
        ref_rnn_object = nn.LSTMCell(input_size, hidden_size, bias=False)

        # copy weights from one rnn to the other
        ref_rnn_object.load_state_dict(quant_rnn_object.state_dict())

        input = torch.randn(batch, input_size)
        hidden = torch.randn(batch, hidden_size)
        cell = torch.randn(batch, hidden_size)

        quant_hout, quant_cout = quant_rnn_object(input, hx=(hidden, cell))
        ref_hout, ref_cout = ref_rnn_object(input, hx=(hidden, cell))

        # The difference between reference and quantized should be bounded in a range
        # Small values which become 0 after quantization lead to large relative errors. rtol and atol could be
        # much smaller without those values
        utils.compare(quant_hout, ref_hout, rtol=1e-4, atol=1e-4)
        utils.compare(quant_cout, ref_cout, rtol=1e-4, atol=1e-4)

        # check that quantization introduces some error
        utils.assert_min_mse(quant_hout, ref_hout, tol=1e-20)
        utils.assert_min_mse(quant_cout, ref_cout, tol=1e-20)

    def test_quant_input_hidden(self, verbose):
        """QuantLSTMCell vs. manual input quantization + pytorchLSTMCell."""
        batch = 15
        input_size = 121
        hidden_size = 51
        num_bits = 4

        quant_desc_input = tensor_quant.QuantDescriptor(num_bits=num_bits)
        quant_desc_weight = tensor_quant.QuantDescriptor(num_bits=num_bits)
        quant_rnn_object = quant_rnn.QuantLSTMCell(input_size, hidden_size, bias=False,
                quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)
        ref_rnn_object = nn.LSTMCell(input_size, hidden_size, bias=False)

        input = torch.randn(batch, input_size)
        hidden = torch.randn(batch, hidden_size)
        cell = torch.randn(batch, hidden_size)

        quant_hout, quant_cout = quant_rnn_object(input, hx=(hidden, cell))

        quant_input, quant_hidden = utils.quantize_by_range_fused((input, hidden), num_bits)

        utils.copy_state_and_quantize_fused(ref_rnn_object, quant_rnn_object, num_bits)

        ref_hout, ref_cout = ref_rnn_object(quant_input, hx=(quant_hidden, cell))

        utils.compare(quant_hout, ref_hout)
        utils.compare(quant_cout, ref_cout)

    def test_quant_input_hidden_bias(self, verbose):
        """QuantLSTMCell vs. manual input quantization + pytorchLSTMCell
            bias should not be quantized
        """
        batch = 9
        input_size = 23
        hidden_size = 31
        num_bits = 7

        quant_desc_input = tensor_quant.QuantDescriptor(num_bits=num_bits)
        quant_desc_weight = tensor_quant.QuantDescriptor(num_bits=num_bits)
        quant_rnn_object = quant_rnn.QuantLSTMCell(input_size, hidden_size, bias=True,
                quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)
        ref_rnn_object = nn.LSTMCell(input_size, hidden_size, bias=True)

        input = torch.randn(batch, input_size)
        hidden = torch.randn(batch, hidden_size)
        cell = torch.randn(batch, hidden_size)

        quant_hout, quant_cout = quant_rnn_object(input, hx=(hidden, cell))

        quant_input, quant_hidden = utils.quantize_by_range_fused((input, hidden), num_bits)

        utils.copy_state_and_quantize_fused(ref_rnn_object, quant_rnn_object, num_bits)

        ref_hout, ref_cout = ref_rnn_object(quant_input, hx=(quant_hidden, cell))

        utils.compare(quant_hout, ref_hout)
        utils.compare(quant_cout, ref_cout)

    def test_quant_different_prec(self, verbose):
        """QuantLSTMCell vs. manual input quantization + pytorch LSTMCell
            different input and weight precisions
        """
        batch = 27
        input_size = 11
        hidden_size = 10
        num_bits_weight = 4
        num_bits_input = 8

        quant_desc_input = tensor_quant.QuantDescriptor(num_bits=num_bits_input)
        quant_desc_weight = tensor_quant.QuantDescriptor(num_bits=num_bits_weight)
        quant_rnn_object = quant_rnn.QuantLSTMCell(input_size, hidden_size, bias=False,
                quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)
        ref_rnn_object = nn.LSTMCell(input_size, hidden_size, bias=False)

        input = torch.randn(batch, input_size)
        hidden = torch.randn(batch, hidden_size)
        cell = torch.randn(batch, hidden_size)

        quant_hout, quant_cout = quant_rnn_object(input, hx=(hidden, cell))

        quant_input, quant_hidden = utils.quantize_by_range_fused((input, hidden), num_bits_input)

        utils.copy_state_and_quantize_fused(ref_rnn_object, quant_rnn_object, num_bits_weight)

        ref_hout, ref_cout = ref_rnn_object(quant_input, hx=(quant_hidden, cell))

        utils.compare(quant_hout, ref_hout)
        utils.compare(quant_cout, ref_cout)


class TestQuantLSTM():
    """
    tests for quant_rnn.QuantLSTM
    default parameters in QuantLSTM:
        bias=True,
        quant_weight=True, bits_weight=8, fake_quantTrue, quant_mode_weight='channel',
        quant_input=True, bits_acts=8, quant_mode_input='tensor'

    Tests of real quantization mode (nonfake) are disabled as it is not fully supported yet.
    """

    def test_basic_forward(self, verbose):
        """Do a forward pass on the layer module and see if anything catches fire."""
        batch = 5
        input_size = 13
        hidden_size = 31
        seq_len = 1

        quant_desc_input = tensor_quant.QuantDescriptor(num_bits=8)
        quant_desc_weight = tensor_quant.QuantDescriptor(num_bits=8, axis=(1,))
        quant_rnn_object = quant_rnn.QuantLSTM(input_size, hidden_size,
                num_layers=1, bias=False, batch_first=False, dropout=0, bidirectional=False,
                quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)
        input = torch.randn(seq_len, batch, input_size)
        hidden = torch.randn(seq_len, batch, hidden_size)
        cell = torch.randn(seq_len, batch, hidden_size)
        quant_rnn_object(input, hx=(hidden, cell))

    def test_no_quant(self, verbose):
        """QuantLSTM with quantization disabled vs. pytorch LSTM."""
        batch = 11
        input_size = 14
        hidden_size = 22
        seq_len = 1
        quant_rnn_object = quant_rnn.QuantLSTM(input_size, hidden_size,
                num_layers=1, bias=False, batch_first=False, dropout=0, bidirectional=False)
        quant_rnn_object._input_quantizers[0].disable()
        quant_rnn_object._weight_quantizers[0].disable()
        ref_rnn_object = nn.LSTM(input_size, hidden_size,
                num_layers=1, bias=False, batch_first=False, dropout=0, bidirectional=False)

        # copy weights from one rnn to the other
        ref_rnn_object.load_state_dict(quant_rnn_object.state_dict())

        input = torch.randn(seq_len, batch, input_size)
        hidden = torch.randn(seq_len, batch, hidden_size)
        cell = torch.randn(seq_len, batch, hidden_size)

        quant_out, (quant_hout, quant_cout) = quant_rnn_object(input)
        ref_out, (ref_hout, ref_cout) = ref_rnn_object(input)

        utils.compare(quant_out, ref_out)
        utils.compare(quant_hout, ref_hout)
        utils.compare(quant_cout, ref_cout)

    def test_no_quant_input_hidden(self, verbose):
        """QuantLSTM with quantization disabled vs. pytorch LSTM for input and hidden inputs."""
        batch = 13
        input_size = 19
        hidden_size = 20
        seq_len = 1

        quant_rnn_object = quant_rnn.QuantLSTM(input_size, hidden_size,
                num_layers=1, bias=False, batch_first=False, dropout=0, bidirectional=False)
        quant_rnn_object._input_quantizers[0].disable()
        quant_rnn_object._weight_quantizers[0].disable()
        ref_rnn_object = nn.LSTM(input_size, hidden_size,
                num_layers=1, bias=False, batch_first=False, dropout=0, bidirectional=False)

        # copy weights from one rnn to the other
        ref_rnn_object.load_state_dict(quant_rnn_object.state_dict())

        input = torch.randn(seq_len, batch, input_size)
        hidden = torch.randn(seq_len, batch, hidden_size)
        cell = torch.randn(seq_len, batch, hidden_size)

        quant_out, (quant_hout, quant_cout) = quant_rnn_object(input, hx=(hidden, cell))
        ref_out, (ref_hout, ref_cout) = ref_rnn_object(input, hx=(hidden, cell))

        utils.compare(quant_out, ref_out)
        utils.compare(quant_hout, ref_hout)
        utils.compare(quant_cout, ref_cout)

    def test_no_quant_all_modes(self, verbose):
        """QuantLSTM with quantization disabled vs. pytorch LSTM for all modes."""

        def testcase(input_size, hidden_size, seq_len, batch, num_layers, bias, batch_first, dropout, bidirectional):

            quant_rnn_object = quant_rnn.QuantLSTM(input_size, hidden_size,
                    num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout,
                    bidirectional=bidirectional)

            num_quantizers = num_layers * 2 if bidirectional else num_layers
            for i in range(num_quantizers):
                quant_rnn_object._input_quantizers[i].disable()
                quant_rnn_object._weight_quantizers[i].disable()

            ref_rnn_object = nn.LSTM(input_size, hidden_size,
                    num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout,
                    bidirectional=bidirectional)

            # copy state from one rnn to the other
            ref_rnn_object.load_state_dict(quant_rnn_object.state_dict())

            input = torch.randn(seq_len, batch, input_size)
            num_directions = 2 if bidirectional else 1
            hidden = torch.randn(num_layers*num_directions, batch, hidden_size)
            cell = torch.randn(num_layers*num_directions, batch, hidden_size)

            quant_out, (quant_hout, quant_cout) = quant_rnn_object(input, hx=(hidden, cell))
            ref_out, (ref_hout, ref_cout) = ref_rnn_object(input, hx=(hidden, cell))

            utils.compare(quant_out, ref_out)
            utils.compare(quant_hout, ref_hout)
            utils.compare(quant_cout, ref_cout)

        # test various permuatations of the following parameters:
        #   size, num_layers, bias, batch_first, dropout, bidirectional
        testcase(32, 27, 1, 1, 1, False, False, 0, False)
        testcase(19, 63, 1, 1, 2, False, False, 0, False)
        testcase(11, 41, 1, 1, 1, True, False, 0, False)
        testcase(33, 31, 1, 1, 1, False, True, 0, False)
        # testcase(32, 32, 1, 1, 2, False, False, 0.5, False) #TODO(pjudd) this fails look into dropout seeding
        testcase(73, 13, 1, 1, 1, False, False, 0, True)

    def test_against_unquantized(self, verbose):
        """Quantization should introduce bounded error utils.compare to pytorch implementation."""
        batch = 21
        input_size = 33
        hidden_size = 25
        seq_len = 1

        quant_desc_input = tensor_quant.QuantDescriptor(num_bits=16)
        quant_desc_weight = tensor_quant.QuantDescriptor(num_bits=16, axis=(1,))
        quant_rnn_object = quant_rnn.QuantLSTM(input_size, hidden_size,
                num_layers=1, bias=False, batch_first=False, dropout=0, bidirectional=False,
                quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)
        ref_rnn_object = nn.LSTM(input_size, hidden_size,
                num_layers=1, bias=False, batch_first=False, dropout=0, bidirectional=False)

        # copy weights from one rnn to the other
        ref_rnn_object.load_state_dict(quant_rnn_object.state_dict())

        input = torch.randn(seq_len, batch, input_size)
        hidden = torch.randn(seq_len, batch, hidden_size)
        cell = torch.randn(seq_len, batch, hidden_size)

        quant_out, (quant_hout, quant_cout) = quant_rnn_object(input, hx=(hidden, cell))
        ref_out, (ref_hout, ref_cout) = ref_rnn_object(input, hx=(hidden, cell))

        # The difference between reference and quantized should be bounded in a range
        # Small values which become 0 after quantization lead to large relative errors. rtol and atol could be
        # much smaller without those values
        utils.compare(quant_out, ref_out, rtol=1e-4, atol=1e-4)
        utils.compare(quant_hout, ref_hout, rtol=1e-4, atol=1e-4)
        utils.compare(quant_cout, ref_cout, rtol=1e-4, atol=1e-4)

        # check that quantization introduces some error
        utils.assert_min_mse(quant_out, ref_out, tol=1e-20)
        utils.assert_min_mse(quant_hout, ref_hout, tol=1e-20)
        utils.assert_min_mse(quant_cout, ref_cout, tol=1e-20)

    def test_quant_input_hidden(self, verbose):
        """QuantLSTM vs. manual input quantization + pytorchLSTM."""
        batch = 13
        input_size = 17
        hidden_size = 7
        seq_len = 1
        num_bits = 6

        quant_desc_input = tensor_quant.QuantDescriptor(num_bits=num_bits)
        quant_desc_weight = tensor_quant.QuantDescriptor(num_bits=num_bits)
        quant_rnn_object = quant_rnn.QuantLSTM(input_size, hidden_size, num_layers=1, bias=False,
                batch_first=False, dropout=0, bidirectional=False,
                quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)
        ref_rnn_object = nn.LSTM(input_size, hidden_size, num_layers=1, bias=False,
                batch_first=False, dropout=0, bidirectional=False)

        input = torch.randn(seq_len, batch, input_size)
        hidden = torch.randn(seq_len, batch, hidden_size)
        cell = torch.randn(seq_len, batch, hidden_size)

        quant_input, quant_hidden = utils.quantize_by_range_fused((input, hidden), num_bits)

        utils.copy_state_and_quantize_fused(ref_rnn_object, quant_rnn_object, num_bits)

        quant_out, (quant_hout, quant_cout) = quant_rnn_object(input, hx=(hidden, cell))
        ref_out, (ref_hout, ref_cout) = ref_rnn_object(quant_input, hx=(quant_hidden, cell))

        utils.compare(quant_out, ref_out)
        utils.compare(quant_hout, ref_hout)
        utils.compare(quant_cout, ref_cout)

    def test_quant_input_hidden_bias(self, verbose):
        """QuantLSTM vs. manual input quantization + pytorchLSTM."""
        batch = 17
        input_size = 13
        hidden_size = 7
        seq_len = 1
        num_bits = 5

        quant_desc_input = tensor_quant.QuantDescriptor(num_bits=num_bits)
        quant_desc_weight = tensor_quant.QuantDescriptor(num_bits=num_bits)
        quant_rnn_object = quant_rnn.QuantLSTM(input_size, hidden_size, num_layers=1, bias=True,
                batch_first=False, dropout=0, bidirectional=False,
                quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)
        ref_rnn_object = nn.LSTM(input_size, hidden_size, num_layers=1, bias=True,
                batch_first=False, dropout=0, bidirectional=False)

        input = torch.randn(seq_len, batch, input_size)
        hidden = torch.randn(seq_len, batch, hidden_size)
        cell = torch.randn(seq_len, batch, hidden_size)

        quant_input, quant_hidden = utils.quantize_by_range_fused((input, hidden), num_bits)

        utils.copy_state_and_quantize_fused(ref_rnn_object, quant_rnn_object, num_bits)

        quant_out, (quant_hout, quant_cout) = quant_rnn_object(input, hx=(hidden, cell))
        ref_out, (ref_hout, ref_cout) = ref_rnn_object(quant_input, hx=(quant_hidden, cell))

        utils.compare(quant_out, ref_out)
        utils.compare(quant_hout, ref_hout)
        utils.compare(quant_cout, ref_cout)

    def test_quant_different_prec(self, verbose):
        """QuantLSTM vs. manual input quantization + pytorchLSTM."""
        batch = 22
        input_size = 23
        hidden_size = 24
        seq_len = 1
        num_bits_weight = 4
        num_bits_input = 8

        quant_desc_input = tensor_quant.QuantDescriptor(num_bits=num_bits_input)
        quant_desc_weight = tensor_quant.QuantDescriptor(num_bits=num_bits_weight)
        quant_rnn_object = quant_rnn.QuantLSTM(input_size, hidden_size, num_layers=1, bias=False,
                batch_first=False, dropout=0, bidirectional=False,
                quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)
        ref_rnn_object = nn.LSTM(input_size, hidden_size, num_layers=1, bias=False,
                batch_first=False, dropout=0, bidirectional=False)

        input = torch.randn(seq_len, batch, input_size)
        hidden = torch.randn(seq_len, batch, hidden_size)
        cell = torch.randn(seq_len, batch, hidden_size)

        quant_input, quant_hidden = utils.quantize_by_range_fused((input, hidden), num_bits_input)

        utils.copy_state_and_quantize_fused(ref_rnn_object, quant_rnn_object, num_bits_weight)

        quant_out, (quant_hout, quant_cout) = quant_rnn_object(input, hx=(hidden, cell))
        ref_out, (ref_hout, ref_cout) = ref_rnn_object(quant_input, hx=(quant_hidden, cell))

        utils.compare(quant_out, ref_out)
        utils.compare(quant_hout, ref_hout)
        utils.compare(quant_cout, ref_cout)


class TestEpilogue():
    """Run after all tests to save globals."""

    def test_save_vectors(self, verbose):
        """Save test vectors to file."""
        if SAVE_VECTORS:
            torch.save(TEST_VECTORS, VECTOR_FILE)
            raise Exception('Saved test vectors to {}, for testing set SAVE_VECTORS = 0'.format(VECTOR_FILE))
