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

#np.finfo(np.float32)
#finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)

class Haptic2AudioRNN(nn.Module):
    def __init__(self):
        super(Haptic2AudioRNN,self).__init__()

        self.input_size = hp_rnn.input_size
        self.hidden_size = hp_rnn.hidden_size
        self.num_layers = hp_rnn.num_layers
        self.batch_size = hp_rnn.batch_size
        self.sequence_len = hp_rnn.sequence_len

        #RNN
#        self.model = nn.RNN(input_size = self.input_size, hidden_size = self.hidden_size,
#                             num_layers = self.num_layers, batch_first = True)
        #LSTM
        self.model = nn.RNN(input_size = self.input_size, hidden_size = self.hidden_size,
                             num_layers = self.num_layers, batch_first = True)
    def forward(self, x, hidden):
#        hidden = None
        x = x.view(self.batch_size, self.sequence_len, self.input_size)
        out, hidden = self.model(x, hidden)
        return hidden, out.view(-1, self.hidden_size)

    def init_hidden(self):
        # Set initial states 
        hidden = None
        #hidden = Variable(torch.zeros(self.num_layers,self.batch_size, self.hidden_size).type(torch.DoubleTensor))

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
        dataSet = dataloader.AudioLoader(os.path.join(self.inPath),os.path.join(self.inPath2))
#        dataSet = dataloader.AudioLoader(self.inPath,self.inPath2)

        trainLoader = torchData.DataLoader(
            dataset = dataSet,
            batch_size = hp_rnn.batch_size,
            shuffle = False
        )

        criterion = nn.KLDivLoss(size_average = False)
        optimizer = torch.optim.Adam(self.model.parameters(),lr = hp_rnn.learning_rate)

        print('Train Start...')

        for epoch in range(hp_rnn.num_epochs):
            
            start_time = time.time()

            for idx, data in enumerate(trainLoader):#,0

                x, y = data

                x = Variable(x)
                
                y = Variable(y.type(torch.DoubleTensor), requires_grad = False)

                optimizer.zero_grad()
                loss = 0
                hidden = self.model.init_hidden()

                hidden, outputs = self.model.forward(x,hidden)

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
            self.save(self.outPath, 'epoch' + str(epoch))
            
        self.save(self.outPath, 'final')
        #print(self.lossHistory)


    def test(self):

        print ('Test Strat...')
        timeText = util.getTime()
        num = 1
        for epoch in range(1,num+1,1):#0,num
            self.load(self.outPath, num = epoch)
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
                                sum_3axis = (sum_3axis - 9) / 4.
                                each_line = sum_3axis#[sum_3axis, each_line[-1]]

                                data_accel[idx2] = each_line

                            output_data = list()
                            result = list()

                            pointer = 0
                            pointer_bool = True
                            for i in range(0,hp_rnn.sequence_len):

                                if pointer < 9:#input_size-1
                                    output_data = data_accel[:pointer+1]
                                    output_data = np.pad(output_data, (hp_rnn.input_size-pointer-1,0),'constant',constant_values=(0))
                                else:
                                    output_data = data_accel[pointer-9:pointer+1]

                                if pointer_bool :
                                    pointer += 2
                                else:
                                    pointer +=3

                                pointer_bool =  not pointer_bool

                                if i == hp_rnn.sequence_len-1:
                                    result.append(result[-1])
                                else:
                                    result.append(output_data)

                            result = np.array(result)
                            result = torch.from_numpy(result)
                            result = Variable(result)
                            
                            hidden = self.model.init_hidden()
                            hidden, outputs = self.model.forward(result,hidden)
                            
                            outputs = util.denorm(outputs)
                            util.printInfo(outputs)
                            
                            outputs = outputs * 2.2
                            outputs = outputs.data.numpy()
                            outputs = np.expm1(outputs)
                            outputs = outputs.T
                                              
                            outputs = np.power (outputs,1.3)

                            audio = stft.griffinLim(outputs)
                            filename='converted_'+os.path.basename(path)+os.path.splitext(f)[0]+'.wav'
                            librosa.output.write_wav("./output_test/"+timeText+str(epoch)+"_"+os.path.splitext(f)[0]+".wav", audio, sr = 16000)
        return 
