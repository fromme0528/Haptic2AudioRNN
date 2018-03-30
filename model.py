
# SimpleRNN

'''

Input Size : 250 (500Hz Accleration)
Output Size : 101,257

101,257

250 -> 101
10->1

np.finfo(np.float32)
finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)

'''


'''
    sequence_length = 28
    input_size = 28
    hidden_size= 128
    num_layers = 2
    num_classes = 10

    num_data = 9
    num_epochs = 3
    batch_size = 1
    learning_rate = 0.0001


2000,2 -> 8000,1로 만들기
(accel, material)-> (s,s,s,s)
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data as torchData
import torch.nn.functional as F
import os
import csv
import pickle
import numpy as np
import hparams as hp
from hparams import Linear as hp_linear
from hparams import Rnn as hp_rnn
from hparams import Default as hp_default
import dataloader
import time
import util
import stft
import gc
import librosa
import soundfile as sf

def Flatten_fun(x):
    N, H, W = x.size()
    return x.view(N, -1)

class Haptic2AudioRNN(nn.Module):
    def __init__(self):
        super(Haptic2AudioRNN,self).__init__()

        self.input_size = hp_rnn.input_size
        self.hidden_size = hp_rnn.hidden_size
        self.num_layers = hp_rnn.num_layers
        self.num_classes = hp_rnn.num_classes
        self.batch_size = hp_rnn.batch_size
        self.sequence_len = hp_rnn.sequence_len

        self.model = nn.RNN(input_size = self.input_size, hidden_size = self.hidden_size,
                             num_layers = self.num_layers, batch_first = True)
        #LSTM

    def forward(self, x, hidden):
        hidden = None
        
        #        print(x) #1x2000x2
        x = x.view(self.batch_size, self.sequence_len, self.input_size)

#        print(hidden) #1x2x8000
        out, hidden = self.model(x, hidden)
        return hidden, out.view(-1, self.num_classes)

    def init_hidden(self):
        # Set initial states 
        hidden = Variable(torch.zeros(self.num_layers,self.batch_size, self.hidden_size))

        #h0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)) 
        #c0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        return hidden
         

class Manager(nn.Module):

    def __init__(self, inPath, inPath2, outPath):

        super(Manager, self).__init__()

        self.inPath = inPath
        self.inPath2 = inPath2
        self.outPath = outPath

        self.model = Haptic2AudioRNN()

#        if torch.cuda.is_available():
#            self.model.cuda()

        self.model.double()

        self.lossHistory = list()

    def load(self,inPath, time = '', num = 1):

        try:
            #load the model files which are created lastly
            files = os.listdir(inPath)
            files = [f for f in files if os.path.splitext(f)[-1] == '.model']
            #files.sort(reverse = True)
            files.sort()
            #timeText = files[0][:10] + '_'
            #self.model = torch.load(os.path.join(inPath, timeText + prefix + 'rnn.model'))
            self.model = torch.load(os.path.join(inPath, files[-num]))
            #if torch.cuda.is_available():
            #    self.model.cuda()

        except:
            print('error : can\'t load model')

        else:
            #print('successfully loaded all model - ',timeText + prefix)
            print('successfully loaded all model - ',files[-num])


    def save(self, outPath, prefix = ''):

        timeText = util.getTime()

        if not prefix == '':

            prefix = prefix + '_'

        try:
            torch.save(self.model.cpu(), os.path.join(outPath, timeText + prefix+'rnn.model'))
        except:
            print('error : can\'t save model')

        else:
            print('successfully saved model - ', timeText + prefix)

    def train(self):

        print('Dataset Load')
        dataSet = dataloader.AudioLoader(os.path.join(self.inPath),os.path.join(self.inPath2))
#        dataSet = dataloader.AudioLoader(self.inPath,self.inPath2)

        trainLoader = torchData.DataLoader(
            dataset = dataSet,
            batch_size = hp_rnn.batch_size,
            shuffle = False
        )

        criterion = nn.KLDivLoss(size_average = True)
        optimizer = torch.optim.Adam(self.model.parameters(),lr = hp_rnn.learning_rate)

        print('Train Start...')

        for epoch in range(hp_rnn.num_epochs):
            
            start_time = time.time()

            for idx, data in enumerate(trainLoader):#,0

                # y : audio(8000)
                x, y = data
                x = Variable(x)# x : accel(2000,2)

                y = Variable(y.type(torch.DoubleTensor), requires_grad = False)

                optimizer.zero_grad()
                loss = 0
                hidden = self.model.init_hidden()

                hidden, outputs = self.model.forward(x,hidden)
                
#                loss = torch.sum(torch.abs(y -  outputs.long())) / 8000.0
                outputs = outputs.view(1,8000)
#                print(outputs)
 #               print(y)
#                loss = criterion(outputs,y)
#                outputs += torch.min(outputs)
#                outputs /= torch.sum(outputs)
                #loss = criterion(F.log_softmax(outputs/torch.sum(outputs),dim=1),F.softmax(y/torch.sum(y),dim=1))
                
                loss = criterion(F.log_softmax(outputs/torch.sum(outputs),dim=1),F.softmax(y/torch.sum(y),dim=1))
                #loss = torch.sum(torch.pow((y -  outputs),2)) / 8000.0
                
                loss.backward()
                optimizer.step()
                #print(loss.data[0])
                if idx%20 == 0:
                    util.printInfo(y)
                    util.printInfo(outputs)
                    print ('Epoch [%d/%d], Iter [%d/%d], Loss: %.8f'
                       %(epoch+1, hp_rnn.num_epochs,idx+1, 332//hp_rnn.batch_size, loss.data[0]))
                    print("--- %s seconds for epoch ---" % (time.time() - start_time))
                self.lossHistory.append((epoch,idx,loss.data[0]))

            self.save(self.outPath, 'epoch' + str(epoch))
            
        self.save(self.outPath, 'final')
        print(self.lossHistory)


    def test(self):

        print ('Test Strat...')
        timeText = util.getTime()
        num = 4
        for i in range(1,+num+1,1):
            self.load(self.outPath, num = i)
            for path, dirs, files in os.walk(self.inPath):

                for f in files:

                    if os.path.splitext(f)[-1] == '.csv':
                        
                        with open(os.path.join(path,f), 'r') as csvfile:

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
                            data_accel = Variable(data_accel)

                            hidden = self.model.init_hidden()
                            hidden, outputs = self.model.forward(data_accel,hidden)
                            outputs = outputs.view(1,8000)
                            outputs = np.expm1(outputs.data.numpy())[0]
                            outputs = np.array(outputs)
                            
                            #util.printInfo(outputs)
                            #librosa.output.write_wav(self.inPath2+"/"+timeText+str(i)+"_"+os.path.splitext(f)[0]+".wav", outputs, sr = 16000, norm = True)
                            sf.write(self.inPath2+"/"+timeText+str(i)+"_"+os.path.splitext(f)[0]+".wav", outputs, 16000)
                            #power for noise
#                            output = np.power(output,1.5)
        return 






