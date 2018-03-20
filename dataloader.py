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

from sklearn.decomposition import PCA

# Problem : Validation Set
# https://github.com/pytorch/examples/blob/master/snli/train.py

class AudioLoader(torchData.Dataset):

    # "input_accel_mmdd"
    # "input_audio_mmdd"

    def __init__(self, inPathAccel, inPathAudio,isShuffle=False):

        files_accel = os.listdir(inPathAccel) 
        files_accel = [f for f in files_accel if os.path.splitext(f)[-1] == '.csv']
        files_accel.sort()
        files_audio = os.listdir(inPathAudio) 
        files_audio = [f for f in files_audio if os.path.splitext(f)[-1] == '.wav']
        files_audio.sort()
        
#        print(files_accel)
#        print(files_audio)
        
        if isShuffle:
            random.shuffle(files_accel)

        self.inPathAccel = inPathAccel
        self.inPathAudio = inPathAudio
        self.len = len(files_accel)
        self.fileList_accel = files_accel[:self.len]
        self.fileList_audio = files_audio[:self.len]
        self.isShuffle = isShuffle

        print('len of dataset : ',self.len, len(files_audio))


    def __getitem__(self, idx):

        with open(os.path.join(self.inPathAccel,self.fileList_accel[idx]), 'r') as csvfile:

            rdr = csv.reader(csvfile)
            data_accel = [line for line in rdr]

            for idx2,each_line in enumerate(data_accel) :

                each_line = [float(i) for i in each_line]
                
                #x,y,z 3 axis -> sum(x,y,z) 1 axis and material property
                sum_3axis = np.sum(each_line[0:2])
                sum_3axis *= 10
                each_line = [sum_3axis, each_line[-1]]

                data_accel[idx2] = each_line

            data_accel = np.array(data_accel)
            data_accel = torch.from_numpy(data_accel)

        #with open(os.path.join(self.inPathAudio, self.fileList_audio[idx]),'rb') as fs:
        
        audio, rate = librosa.load(self.inPathAudio+'/'+self.fileList_audio[idx], mono=True, sr = hp_default.sr) 
#        여기문제~~안읽힘~~
        #Problem : Audio Preprocessing
        #audio_normalized = preprocessing.normalizeAudio(audio)
        #audio = processing(audio, mode = 'pre', input_type = 'audio')
        #audio = [100*a for a in audio]
        
        data_audio = torch.from_numpy(np.array(audio))

        return data_accel, data_audio #input-label

    def __len__(self):
        return self.len

def processing(input_list, mode, input_type):
    
    if mode == 'pre':
        if input_type == 'accel':
            input_list = 10 * input_list
        elif input_type =='audio':
            input_list = 10 * input_list
    
    elif mode =='post':
        if input_type == 'audio':
            input_list = input_list / 10.0

    return input_list

# filePath = "acceleration_0213\wood_hit.csv"
# 500
def divide_accel_csv(inPath):

    # 'cp949' codec can't decode
    with open(os.path.join(inPath),'r',encoding='UTF8') as csvfile:

        data_accel = csv.reader(csvfile)
        tmp = 400#400 for plastic, 375 for steel
        result = list()
        for idx, data in enumerate(data_accel):
            if idx<tmp:
                continue
            if idx>tmp+249:
                with open (os.path.join("dataset/accel_split/accel_plastic_"+str(int((idx)/500)-1)+'.csv'),'w',newline='') as fs:
                    wr = csv.writer(fs)
                    for row in result:
                        wr.writerow(row)
                result = []
                tmp = tmp+500
                #print(str(int((idx+10)/50)))
                continue

                #label
            new = data[:]
            new.append('0.0')
            
            result.append(new)

def duplicate(inPath_folder):

    files_accel = os.listdir(inPath_folder) 
    files_accel = [f for f in files_accel if os.path.splitext(f)[-1] == '.csv']

    for inPath in files_accel:
    # 'cp949' codec can't decode
        print(inPath_folder + inPath)
        with open(os.path.join(inPath_folder + inPath),'r',encoding='UTF8') as csvfile:

            data_accel = csv.reader(csvfile)
            data_accel = [line for line in data_accel]

            output = list()

            for idx,line in enumerate(data_accel) :

                line = [float(i) for i in line]

                if idx != (len(data_accel) - 1):
                    line_next = [float(i) for i in data_accel[idx+1]]
#                    interval = list(set(data_accel[idx+1][0:2]) - set(line[0:2]))
                    interval = list()
                    interval.append(line_next[0] - line[0])
                    interval.append(line_next[1] - line[1])
                    interval.append(line_next[2] - line[2])

                    for i in range(0,8,1):
                        temp = list()
                        temp.append(line[0] + (interval[0] * i * 0.125))
                        temp.append(line[1] + (interval[1] * i * 0.125))
                        temp.append(line[2] + (interval[2] * i * 0.125))
                        temp.append(line[-1])
                        output.append(temp)
                else:
                    for i in range(0,8,1):
                        output.append(line)
        with open (os.path.join("dataset/input_accel/"+inPath),'w',newline='') as fs:
            wr = csv.writer(fs)
            for row in output:
                wr.writerow(row)
                

#duplicate("./dataset/accel_split/")


                

# input : a,b
# output : data in (b,c)

def interpolation(aList,bList,num):

    outputList = list()

    interval = aList[0:2] - bList[0:2]

    for i in range(1,num):

        output.append(b + interval/num)

    return outputList




#divide_accel_csv("dataset/accel_plastic.csv")

'''
PCA (x,y,z -> 1 axis)
input list [[x1,y1,z1], [x2,y2,z2], ...]
input shape (N,3)
output shape (N,1)
'''

def PCA(input_list):
    
    pca = PCA(n_components=3)
    
    pca.fit(input_list)
    
    return pca.singular_values_


#inPath = "wav_0213\water_hit_volumeUp.wav"
#16000
def divide_audio(inPath):

    #load audio#hp_default.sr
    audio, rate = librosa.load(inPath, mono=True, sr = hp_default.sr)

    print (audio, rate)
    print (len(audio))
    print (len(audio)/rate)
    print (int(len(audio)/rate))

    #Set Starting Point
    temp = 12800
    for i in range(0,int(len(audio)/rate)+1):
        librosa.output.write_wav(os.path.join("dataset/audio_split","audio_steel_"+str(i)+".wav"),audio[int(temp):int(temp+8000)],sr=hp_default.sr)
        temp += 16000

#divide_audio("dataset/audio_plastic_all.wav")
#divide_audio("dataset/audio_steel_all.wav")

def divide_accel_csv_old(inPath):
    with open(os.path.join(inPath),'r') as csvfile:
        data_accel = csv.reader(csvfile)
        tmp = 40
        blankCatch = [0,0,0,0]
        result = list()
        for idx, data in enumerate(data_accel):
            if idx<tmp:
                continue
            if idx>tmp+24:
                with open (os.path.join("input_accel_0213\wood_accel_"+str(int((idx+10)/50))+'.csv'),'w',newline='') as fs:
                    wr = csv.writer(fs)
                    for row in result:
                        wr.writerow(row)
                result = []
                tmp = tmp+50
                #print(str(int((idx+10)/50)))
                continue
            # 손으로 보정
            # 연속으로 비어있는 칸도 있고 그럼.
            
            for i in range(1,4):

                if data[i] == "":
                    print(idx)
            new = data[1:]
            new.append('0.0')
            
            result.append(new)
            #result.append(data[1:])

def divide_audio_old():
    #load audio#hp_default.sr
    audio, rate = librosa.load("wav_0213\water_hit_volumeUp.wav", mono=True, sr = hp_default.sr)
    temp = 4800
    for i in range(1,146):
        librosa.output.write_wav(os.path.join("input_audio_0213","sample_audio_"+str(i)+".wav"),audio[int(temp):int(temp+8000)],sr=hp_default.sr)
        temp += 16000
    audio, rate = librosa.load("input_audio_0213\sample_audio_1.wav")
    print(audio)
    print(rate)
    