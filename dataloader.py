import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data as torchData
import os
import csv
import pickle
import numpy as np
import hparams as hp
from hparams import Linear as hp_linear
from hparams import Cnn as hp_cnn
from hparams import Default as hp_default
import time
import util
import stft
import gc
import librosa

class AudioLoader(torchData.Dataset):

    # "input_accel_mmdd"
    # "input_spectro_mmdd"

    def __init__(self, inPathAccel, inPathAudio,size,isShuffle=False):

        files_accel = os.listdir(inPathAccel) 
        files_accel = [f for f in files_accel if os.path.splitext(f)[-1] == '.csv']
        files_audio = os.listdir(inPathAudio) 
        files_audio = [f for f in files_audio if os.path.splitext(f)[-1] == '.wav']

        if isShuffle:
            random.shuffle(files_accel)

        self.inPathAccel = inPathAccel
        self.inPathAudio = inPathAudio
        self.fileList_accel = files_accel[:size]
        self.fileList_audio = files_audio[:size]
        self.len = size
        self.isShuffle = isShuffle

    def __getitem__(self, idx):

        with open(os.path.join(self.inPathAccel,self.fileList_accel[idx]), 'r') as csvfile:

            rdr = csv.reader(csvfile)
            data_accel = [line for line in rdr]

            for idx2,each_line in enumerate(data_accel) :
                each_line = [float(i) for i in each_line]
                data_accel[idx2] = each_line

            data_accel = np.array(data_accel)
            data_accel = torch.from_numpy(data_accel)

        with open(os.path.join(self.inPathAudio, self.fileList_audio[idx]),'rb') as fs:
            
            audio, rate = librosa.load(fs, mono=True, sr = hp_default.sr) 
            
            #Problem : Audio Preprocessing
            #audio_normalized = preprocessing.normalizeAudio(audio)
            
            data_audio = torch.from_numpy(audio_normalized)

        return data_accel, data_audio #input-label

    def __len__(self):
        return self.len
