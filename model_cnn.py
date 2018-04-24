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
from hparams import Cnn as hp_cnn
from hparams import Cnn as hp_rnn
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

#np.finfo(np.float32)
#finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)


#500,1 -> 101,257
class Flatten(nn.Module):
    # from cs231 assignment
    def forward(self, x):
        N, H, W = x.size()
        return x.view(N, -1)

class Haptic2AudioCNN(nn.Module):
    def __init__(self):
        super(Haptic2AudioCNN,self).__init__()

        self.model = nn.Sequential( # (250,1)
            nn.Conv1d(1,16,3,stride = 1, padding = 1),#(25,16)
            nn.BatchNorm1d(16),
            nn.ReLU(inplace = True),
            nn.Conv1d(16,64,3,stride =2, padding = 1),#(13,64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            Flatten(),
            nn.Linear(8000,1024*8, bias = True), #(832 -> 8192)
            nn.ReLU(inplace = True),
            nn.Linear(1024*8,101*257, bias = True), #(8192 -> 25957)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(1,1,250)
        y = self.model(x)
        return y

class Manager(nn.Module):

    def __init__(self, inPath, inPath2, outPath):

        super(Manager, self).__init__()

        self.inPath = inPath
        self.inPath2 = inPath2
        self.outPath = outPath

        self.model = Haptic2AudioCNN()

#        if torch.cuda.is_available():
#            self.model.cuda()

        self.model.double()
        self.lossHistory = list()

    def load(self,inPath, time = '', num = 1):

        try:

            #load the model files which are created lastly
            files = os.listdir(inPath)
            files = [f for f in files if os.path.splitext(f)[-1] == '.model']
            files.sort()
            #timeText = files[0][:10] + '_'
            #self.model = torch.load(os.path.join(inPath, timeText + prefix + 'rnn.model'))
            self.model.load_state_dict(torch.load(os.path.join(inPath, files[-num])))
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
            torch.save(self.model.state_dict(), os.path.join(outPath, timeText + prefix+'rnn.model'))
        except:
            print('error : can\'t save model')

        else:
            print('successfully saved model - ', timeText + prefix)

    def train(self):

        print('Dataset Load')
        dataSet = dataloader.AudioLoaderLinear(os.path.join(self.inPath),os.path.join(self.inPath2))
#        dataSet = dataloader.AudioLoader(self.inPath,self.inPath2)

        trainLoader = torchData.DataLoader(
            dataset = dataSet,
            batch_size = hp_rnn.batch_size,
            shuffle = False
        )

        criterion = nn.KLDivLoss(size_average = False)
        optimizer = torch.optim.Adam(self.model.parameters(),lr = hp_rnn.learning_rate)

        print('Train Start...')

        for epoch in range(10):#hp_rnn.num_epochs
            
            start_time = time.time()

            for idx, data in enumerate(trainLoader):#,0

                x, y = data

                x = Variable(x)
                
                y = Variable(y.type(torch.DoubleTensor), requires_grad = False)

                optimizer.zero_grad()
                loss = 0
                
                print(x)
                print(y)
                outputs = self.model.forward(x)

                y = y.view(1,-1)
                outputs = outputs.view(1,-1)
                outputs = util.denorm(outputs)

                loss = torch.mean((y-outputs)**2) + criterion(F.log_softmax(outputs),F.softmax(y))
                
                loss.backward()
                optimizer.step()
                
                if idx%20 == 0:
                    # util.printInfo(y)
                    # util.printInfo(outputs)
                    print ('Epoch [%d/%d], Iter [%d/%d], Loss: %.8f'
                       %(epoch+1, hp_rnn.num_epochs,idx+1, 332//hp_rnn.batch_size, loss.data[0]))
                    print("--- %s seconds for epoch ---" % (time.time() - start_time))
                    print("loss1 :",torch.mean((y -  outputs)**2).data[0])
                    print("loss2 :",criterion(F.log_softmax(outputs.view(1,-1)), F.softmax(y.view(1,-1))).data[0])

#                self.lossHistory.append((epoch,idx,loss.data[0]))
                gc.collect()
#            self.save(self.outPath, 'epoch' + str(epoch))
            
        self.save(self.outPath, 'final')
        #print(self.lossHistory)


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

                            outputs = self.model.forward(data_accel)
                            outputs = outputs.view(1,8000)
                            outputs = np.expm1(outputs.data.numpy())[0]
                            outputs = np.array(outputs)
                            
                            #util.printInfo(outputs)
                            #librosa.output.write_wav(self.inPath2+"/"+timeText+str(i)+"_"+os.path.splitext(f)[0]+".wav", outputs, sr = 16000, norm = True)
                            sf.write(self.inPath2+"/"+timeText+str(i)+"_"+os.path.splitext(f)[0]+".wav", outputs, 16000)
                            #power for noise
#                            output = np.power(output,1.5)
        return 
