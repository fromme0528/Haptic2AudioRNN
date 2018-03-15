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

#Problem
class Flatten(nn.Module):
    # from cs231 assignment
    def forward(self, x):
        N, H, W = x.size()
        return x.view(N, -1)

def Flatten_fun(x):
    N, H, W = x.size()
    return x.view(N, -1)

#RNN Model

# Manager
class Manager(nn.Module):

    def __init__(self, inPath, inPath2, outPath, dataNum):
        super(Manager,self).__init__()

        self.inPath = inPath
        self.inPath2 = inPath2
        self.outPath = outPath
        self.dataNum = dataNum

        self.model = Accel2WavRNN()
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.double()

        #Probelm : Plotting Loss history
        self.lossHistory = list()

    def load(self,inPath, prefix = '', time = ''):

        if not prefix == '':
            prefix = prefix + '_'

        try:
            #load the model files which are created lastly
            files = os.listdir(inPath)
            files = [f for f in files if os.path.splitext(f)[-1] == '.model']
            files.sort(reverse = True)
            timeText = files[0][:10] + '_'
            self.model = torch.load(os.path.join(inPath, timeText + prefix + 'rnn.model'))
            if torch.cuda.is_available():
                self.model.cuda()
        except:
            print('error : can\'t load model')
        else:
            print('successfully loaded all model - ',timeText + prefix)

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

        dataSet = AudioLoader(os.path.join(self.inPath),os.path.join(self.inPath2),hp_cnn.num_data)

        trainLoader = torchData.DataLoader(
            dataset = dataSet,
            batch_size = hp_cnn.batch_size,
            shuffle = False
        )

        criterion_cnn = nn.CrossEntropyLoss()
        optimizer_cnn = torch.optim.Adam(self.cnn.parameters(), lr=hp_cnn.learning_rate)
        
        print('Train started')
        for epoch in range(hp_cnn.num_epochs):
            for idx, data in enumerate(trainLoader,0):
                start_time = time.time()
                x, y = data # x : accel (25,4), y : spectro (101,257)
                x = Variable(x) 
                x = x.view(hp_cnn.batch_size,4,25)
                
                y = Variable(y.type(torch.FloatTensor), requires_grad = False)
                y = Flatten_fun(y)
                y = y.double()
                optimizer_cnn.zero_grad()
                
                outputs = self.cnn.forward(x)
                
                #loss = criterion_linear(outputs,y)
                
                loss = torch.sum(torch.abs(torch.log(y + 1) - outputs )) / (101.0 * 257.0)
                #loss = crss(
                loss.backward()
                optimizer_cnn.step()
                self.lossHistory.append((epoch,idx,loss.data[0]))
                print ('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                               %(epoch+1, hp_cnn.num_epochs,idx+1, hp_cnn.num_data//hp_cnn.batch_size, loss.data[0]))
                print("--- %s seconds for epoch ---" % (time.time() - start_time))

                gc.collect()

            self.save(self.outPath, 'epoch' + str(epoch))
        self.save(self.outPath, 'final')
        print(self.lossHistory)

    def test(self, prefix):
        
        self.load(self.outPath, prefix = prefix)
#        self.load(self.outPath, prefix = 'epoch0')
        print ("test strat")
        for path, dirs, files in os.walk(self.inPath):
            print (path,files)
            for f in files:
                if os.path.splitext(f)[-1] == '.csv':
                    with open(os.path.join(path,f), 'r') as csvfile:
                        print(f)
                        rdr = csv.reader(csvfile)
                        data_accel = [line for line in rdr]
                        for idx2,each_line in enumerate(data_accel) :
                            each_line = [float(i) for i in each_line]
                            data_accel[idx2] = each_line
                        data_accel = np.array(data_accel)
                        data_accel = Variable(torch.from_numpy(data_accel), requires_grad = False)
                        data_accel = data_accel.contiguous()

                        data_accel = data_accel.view(1,4,25)
                        output = self.cnn.forward(data_accel)
                        output = output.data.numpy()
                        output = output.reshape(101,257)
                        #expm1
                        output = np.exp(output) - 1
                        
                        #power for noise
                        output = np.power(output,1.5)
                        outFile = 'spectro_' + '_' + os.path.splitext(f)[0] + '.pickle'
                        with open(os.path.join(self.inPath2,outFile),'wb') as fs:
                            pickle.dump(output,fs)
                        preprocessing.test(output, self.inPath2,f)
        print('test ended')

#test('cnn')

#Linear Model
class SimpleLinear(nn.Module):

    def __init__(self):
        super(SimpleLinear,self).__init__()
        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(75, 1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024, 1024*8),
            nn.ReLU(inplace = True),
            nn.Linear(1024*8, 101*257),
            nn.Sigmoid()
            )

    def forward(self, x):
        y = self.model(x)
        return y

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

'''

class SimpleRnn(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRnn, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        #Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

#view(-1,)