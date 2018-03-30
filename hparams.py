# -*- coding: utf-8 -*-


# HyperParameters for RNN
# retrieved from yunjei's pytorch tutorial
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py
# GRU is good for small data <-> LSTM. http://aikorea.org/blog/rnn-tutorial-4/


class Rnn:
	sequence_len = 101#2000#8000#2000 #1
	input_size = 10 #2
	hidden_size= 1 #4#4 #8000
	num_layers = 2
	num_classes = 1

	num_epochs = 15
	batch_size = 1
	learning_rate = 0.0001

# Hyper Parameters for audio processing
# hparams from Deep Voice Conversion
# https://github.com/andabi/deep-voice-conversion/blob/master/hparams.py
class Default:
	sr = 16000
	frame_shift = 0.005 #seconds 
	frame_length = 0.025 #seconds
	n_fft= 512

	hop_length = int(sr*frame_shift) #80 
	win_length = int(sr*frame_length) #400
	timeLength = 0.5 #seconds

# Hyper Parameters for Linear
class Linear:
	num_data=9
	num_epochs = 3
	batch_size = 1
	learning_rate = 0.00005

# Hyper Parameters for CNN
class Cnn:
	num_data=20#290#300
	num_epochs = 3#3
	batch_size = 1
	learning_rate = 0.0001#0.00001

