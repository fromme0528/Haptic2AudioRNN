import os
import sys
import argparse
#https://docs.python.org/3/howto/argparse.html
import stft
from model import Manager
from hparams import Linear as hp_linear
from hparams import Cnn as hp_cnn

import librosa
import numpy as np
#import matplotlib.pyplot as plt

# Usage : python main.py <inPath> <outPath> <modelPath> <mode> <prefix>
#
#         python solver.py ./dataset/input_accel ./dataset/audio_split ./output_model train
#         python solver.py ./dataset/test_accel ./output_test ./output_model convert

def main(config):
    model = Manager(config.inPath, config.inPath2, config.modelPath)
    if config.mode == "train":

        model.train()

    elif config.mode == 'convert':

        #model.test(config.prefix)
        model.test()

    else:
        print('Error : Mode must be "train" or "test"')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('inPath', type=str, help = "input acceleration dictionary")
    parser.add_argument('inPath2', type = str, help = "wav label dictionary for train or output dictionary for test")
    parser.add_argument('modelPath', type = str, help = "model Path")
    parser.add_argument('mode', type = str, choices=['train','convert'], help = 'Mode option : train, convert')
    #parser.add_argument('prefix',type = str, help = "final, epoch0~N", default = "final")

    config = parser.parse_args()

    print(config)
    main(config)
