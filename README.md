# Build Transformers from scratch using Attention Mechanism, Pytorch, TensorFlow
I have dealt with this code (from different resources included papers, notebooks, and etc.), wrote and followed its instructions because it was deep training of how these techniques work, and I shared it with fine illustrations for those who want to check it, and take my insights and assumptions of different resources in one notebook, that will help to edit and good understanding.

**Hoping to update it with a better understanding and better techniques in the upcoming times.**

## Table of Contents
* [Introduction](#introduction)
* [Dataset General info](#dataset-general-info)
* [Evaluation](#evaluation)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Run Example](#run-example)

## Introduction
In this tutorial, I will explain, with support, the implement of Attention Mechanism, and transformers in "Attention is all you need paper" from scratch using Pytorch, and TensorFlow. Basically transformer have an encoder-decoder architecture. It is common for language translation models.

## Dataset General info
**General info about the dataset:**
All of the implementations are random numbers generate using random methods. But the implements of transformers from scratch uses:
[TFDS](https://www.tensorflow.org/datasets) to load the [Portugese-English translation dataset](https://github.com/neulab/word-embeddings-for-nmt) from the [TED Talks Open Translation Project](https://www.ted.com/participate/translate).

This dataset contains approximately 50000 training examples, 1100 validation examples, and 2000 test examples.

## Evaluation

The following steps are used for evaluation (tensorflow implements):

* Encode the input sentence using the Portuguese tokenizer (`tokenizer_pt`). Moreover, add the start and end token so the input is equivalent to what the model is trained with. This is the encoder input.
* The decoder input is the `start token == tokenizer_en.vocab_size`.
* Calculate the padding masks and the look ahead masks.
* The `decoder` then outputs the predictions by looking at the `encoder output` and its own output (self-attention).
* Select the last word and calculate the argmax of that.
* Concatentate the predicted word to the decoder input as pass it to the decoder.
* In this approach, the decoder predicts the next word based on the previous words it predicted.

## Technologies
* Programming language: Python.
* Libraries: numpy, python, Time, matplotlib, scipy, ipython, tensorflow-datasets, tensorflow. 
* Application: Jupyter Notebook.

## Setup
To run this project setup the following libraries on your local machine using pip on the terminal after installing Python:\
'''\
pip install numpy\
pip install matplotlib\
pip install scipy\
pip install ipython\
pip install tensorflow-datasets\
pip install TIME-python\
pip install tensorflow\
'''\
To install these packages with conda run:\
'''\
conda install -c anaconda numpy\
conda install -c conda-forge matplotlib\
conda install -c anaconda scipy\
conda install -c anaconda ipython\
conda install -c anaconda tensorflow-datasets\
conda install -c conda-forge tensorflow\
conda install -c conda-forge time\
'''

## Features
* I present to you my illustration notebook of solving transformers mechanism, so from here I combined alot of resources (papers, notebooks, and codes) to learn from it, and so it ended with this shape.

### To Do:
**Briefly about the process of the project work, here are (some) insights that I took care of it:**
1. The Attention Mechanism
2. The General Attention Mechanism
3. The General Attention Mechanism with NumPy and SciPy
4. Implementing Transformers From Scratch Using Pytorch\
    4.1. Introduction\
    4.2. Import libraries\
    4.3. Basic components\
        4.3.1. Create Word Embeddings\
        4.3.2. Positional Encoding\
        4.3.3. Self Attention\
    4.4. Encoder\
    4.5. Decoder\
    4.6. Testing our code
5. Implementing Transformers From Scratch Using TensorFlow\
    5.1. Introduction\
    5.2. Import libraries\
    5.3. Positional encoding\
    5.4. Scaled dot product attention\
    5.5. Multi-head attention\
    5.6. Point wise feed forward network\
    5.7. Encoder and decoder\
        5.7.1. Encoder layer\
        5.7.2. Decoder layer\
        5.7.3. Encoder\
        5.7.4. Decoder\
    5.8. Create the Transformer\
    5.9. Set hyperparameters\
    5.10. Optimizer\
    5.11. Loss and metrics\
    5.12. Training and checkpointing\
    5.13. Evaluate

## Run Example

To run and show analysis, insights, correlation, and results between any set of features of the dataset, here is a simple example of it:

* Note: you have to use a jupyter notebook to open this code file.

1. Run the importing stage.

2. Load the dataset in case of the tensorflow implemet section.

3. Load notebook images zip file.

4. Select which cell you would like to run and show its output.

5. Run Selection/Line in Python Terminal command (Shift+Enter).
