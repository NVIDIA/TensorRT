# Neural Machine Translation (NMT) Using A Sequence To Sequence (seq2seq) Model


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    * [Encoding and embedding](#encoding-and-embedding)
    * [Attention mechanisms](#attention-mechanisms)
    * [Beam search and projection](#beam-search-and-projection)
	* [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleNMT, demonstrates the implementation of Neural Machine Translation (NMT) based on a TensorFlow seq2seq model using the TensorRT API. The TensorFlow seq2seq model is an open sourced NMT project that uses deep neural networks to translate text from one language to another language.

Specifically, this sample is an end-to-end sample that takes a TensorFlow model, builds an engine, and runs inference using the generated network. The sample is intended to be modular so it can be used as a starting point for your machine translation application.

This sample implements German to English translation using the data that is provided by and trained from the [TensorFlow NMT (seq2seq) Tutorial](https://github.com/tensorflow/nmt.git).

Note: Please note that the sample supports Linux only. Windows users can use Windows Subsystem for Linux (WSL) to run sampleNMT.

## How does this sample work?

The basic architecture of the NMT model consists of two sides: an encoder and a decoder. Incoming sentences are translated into sequences of words in a fixed vocabulary. The incoming sequence goes through the **encoder** and is transformed by a network of Recurrent Neural Network (RNN) layers into an internal state space that represents a language-independent "meaning" of the sentence. The **decoder** works the opposite way, transforming from the internal state space back into a sequence of words in the output vocabulary.

### Encoding and embedding

The encoding process requires a fixed vocabulary of words from the source language. Words not appearing in the vocabulary are replaced with an `UNKNOWN` token. Special symbols also represent `START-OF-SENTENCE` and `END-OF-SENTENCE`. After the input is finished, a `START-OF-SENTENCE` is fed in to mark the switch to decoding. The decoder will then produce the `END-OF-SENTENCE` symbol to indicate it is finished translating.

Vocabulary words are represented as word vectors of a fixed size. The mapping from vocabulary word to embedding vector is learned during training.

### Attention mechanisms

Attention mechanisms sit between the encoder and decoder and allow the network to focus on one part of the translation task at a time. It is possible to directly connect the encoding and decoding stages but this would mean the internal state representing the meaning of the sentence would have to cover sentences of all possible lengths at once.

This sample implements [Luong attention](https://arxiv.org/abs/1508.04025). In this model, at each decoder step the target hidden state is combined with all source states using the attention weights. A scoring function weighs each contribution from the source states. The attention vector is then fed into the next decoder stage as an input.

### Beam search and projection

There are several ways to organize the decode stage. The output of the RNN layer is not a single word. The simplest method is to choose the most likely word at each time step, assume that is the correct output, and continue until the decoder generates the `END-OF-SENTENCE` symbol.

A better way to perform the decoding is to keep track of multiple candidate possibilities in parallel and keep updating the possibilities with the most likely sequences. In practice, a small fixed size of candidates works well. This method is called beam search. The beam width is the number of simultaneous candidate sequences that are in consideration at each time step.

As part of beam search we need a mechanism to convert output states into probability vectors over the vocabulary. This is accomplished with the projection layer using a fixed dense matrix.

For more information related to sampleNMT, see [Creating A Network Definition In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#network_c), [Working With Deep Learning Frameworks](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_model), and  [Enabling FP16 Inference Using C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#enable_fp16_c).

### TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Constant layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#constant-layer)
The Constant layer outputs a tensor with values provided as parameters to this layer, enabling the convenient use of constants in computations. As used in the `slp_attention.cpp`, `slp_embedder.cpp` and `slp_projection.cpp` files.

[Gather layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#gather-layer)
The Gather layer implements the `gather` operation on a given axis.  As used in the `slp_embedder.cpp` file.

[MatrixMultiply layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#matrixmultiply-layer)
The MatrixMultiply layer implements matrix multiplication for a collection of matrices.  As used in the `context.cpp`, `multiplicative_alignment.cpp`, `slp_attention.cpp` and `slp_projection.cpp` files.

[RaggedSoftMax layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#raggedsoftmax-layer)
The Ragged SoftMax layer applies the SoftMax function on an input tensor of sequences across the sequence lengths specified by the user.  As used in the `context.cpp` file.

[RNNv2 layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#rnnv2-layer)
The RNNv2 layer implements recurrent layers such as Recurrent Neural Network (RNN), Gated Recurrent Units (GRU), and Long Short-Term Memory (LSTM). It performs a recurrent operation, where the operation is defined by one of several well-known recurrent neural network (RNN) "cells".  As used in the `lstm_encoder.cpp` and `lstm_decoder.cpp` files.

[Shuffle layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#shuffle-layer)
The Shuffle layer implements a reshape and transpose operator for tensors.  As used in the `lstm_encoder.cpp` and `lstm_decoder.cpp` files.

[TopK layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#topk-layer)
The TopK layer finds the top K maximum (or minimum) elements along a dimension, returning a reduced tensor and a tensor of index positions.  As used in the `softmax_likelihood.cpp` file.

## Prerequisites

The model was trained on the [German to English (De-En) dataset](https://github.com/tensorflow/nmt#wmt-german-english) in the WMT database. Before you can run the sample, you need trained model weights and the text and vocabulary data for performing inference.

Run the following command from the `<TensorRT root directory>`. This will download the pre-trained weights, a vocabulary file and an example input text file. In addition, it will preprocess the input text file so that sampleNMT can translate it. The following command prepares all necessary input data.
`./samples/sampleNMT/get_newstest2015.sh`

## Running the sample

Now that you have trained weights, downloaded the text and vocabulary data, and compiled the sample you can run the sample.

1.  Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleNMT` directory. The binary named `sample_nmt` will be created in the `<TensorRT root directory>/bin` directory.
	```
	cd <TensorRT root directory>/samples/sampleNMT
	make
	```

	Where `<TensorRT root directory>` is where you installed TensorRT.

2.  Run the sample to generate the example translation from German to English:
	```
	sample_nmt --data_writer=text
	```

	**Note:** If your data is not located in `<path_to_tensorrt>/data/samples/nmt/deen`, use the `--data_dir=<path_to_data_directory>` option. Where `<path_to_data_directory>` is the path to your data directory. For example:
    ```
    sample_nmt --data_dir=<path_to_data_directory> --data_writer=text
    ```

	The files in the `data` directory contain hardcoded names. Therefore, if you want to translate a different input file, rename the input file to `newstest2015.tok.bpe.32000.de` and put it in the data directory.

	The translated output is located in the `./translation_output.txt` file.

3.  Run the sample to get the BLEU score (the quality of the translated text) for the first 100 sentences:
	```
	sample_nmt --max_inference_samples=100 --data-writer=bleu
	```

4.  Verify your translated output.
		a. Compare your translated output to the `<path_to_tensorrt>/data/newstest2015.tok.bpe.32000.en` translated output file in the TensorRT package.
		b. Compare the quality of your translated output with the 25.85 BLEU score quality metric file in the TensorRT package.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.  For example:
```
data_dir: /workspace/tensorrt/samples/sampleNMT/data/deen
data_writer: text
Component Info:
– Data Reader: Text Reader, vocabulary size = 36548
– Input Embedder: SLP Embedder, num inputs = 36548, num outputs = 512
– Output Embedder: SLP Embedder, num inputs = 36548, num outputs = 512
– Encoder: LSTM Encoder, num layers = 2, num units = 512
– Decoder: LSTM Decoder, num layers = 2, num units = 512
– Alignment: Multiplicative Alignment, source states size = 512, attention keys size = 512
– Context: Ragged softmax + Batch GEMM
– Attention: SLP Attention, num inputs = 1024, num outputs = 512
– Projection: SLP Projection, num inputs = 512, num outputs = 36548
– Likelihood: Softmax Likelihood
– Search Policy: Beam Search Policy, beam = 5
– Data Writer: Text Writer, vocabulary size = 36548
End of Component Info
```

## Additional resources

The following resources provide a deeper understanding about Neural Machine Translation and seq2seq models:

**NMT**
- [Luong, Cho, Manning, (2016)](https://sites.google.com/site/acl16nmt/)
- [Luong, (2016)](https://github.com/lmthang/thesis)
- [Neubig, (2017)](https://arxiv.org/abs/1703.01619)

**Models**
- [OpenNMT](http://opennmt.net/OpenNMT/)
- [NMT (seq2seq) Tutorial](https://github.com/tensorflow/nmt)

**Blogs**
- [Neural Machine Translation Inference in TensorRT](https://devblogs.nvidia.com/neural-machine-translation-inference-tensorrt-4/)
- [Introduction to NMT](https://devblogs.nvidia.com/introduction-neural-machine-translation-with-gpus/)

**Videos**
- [Optimizing NMT with TensorRT](http://on-demand.gputechconf.com/gtc/2018/video/S8822/)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

## Changelog

June 2019
This is the first release of the `README.md` file and sample.

## Known issues

If you would like to train your own weights through the TensorFlow implementation, you can use the `chptToBin.py` script to convert weights in a format that is readable by TensorRT. However, the `chptToBin.py` script may be outdated.
