# SimpleRNN

'''

Input Size : 2000,2 (500Hz Accleration)
Output Size : 8000 (16000Hz Audio Sampling Rate)

Input Size : 2000,2
Output Size : 8000

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
sequence_len = 1
input_size = 2

hidden_size = 8000
num_layers = 2
num_classes = 4

batch_size = 50
num_epochs = 3
learning_rate = 0.001


class Haptic2WaveRNN(nn.Module):
    def __init__(self):
        super(Haptic2WaveRNN,self).__init__()

        self.input_size = hp_rnn.input_size
        self.hidden_size = hp_rnn.hidden_size
        self.num_layers = hp_rnn.num_layers
        self.num_classes = hp_rnn.num_classes

        self.model = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        

    def forward(self, hideen, x):

        x = x.view(batch_size, sequence_len, input_size)
        out, hidden = self.model(x, hidden)
        return hidden, out.view(-1, num_classes)

    def init_hidden(self):
        return Variable(torch.zeros(batch_size, num_layers, hidden_size))


class Haptic2WaveRNN(nn.Module):
    def __init__(self, inPath, inPath2, outPath, num_data):

        super(Haptic2WaveRNN, self).__init__()

        self.inPath = inPath
        self.inPat2 = inPath2
        self.outPath = outPath
        self.num_data = num_data

        self.model = Haptic2WaveRNN()

    if torch.cuda.is_available():
            self.model.cuda()
        self.model.double()

        self.LossHistory = list()

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

        dataSet = AudioLoader(os.path.join(self.inPath),os.path.join(self.inPath2),hp_rnn.num_data)

        trainLoader = torchData.DataLoader(
            dataset = dataSet,
            batch_size = hp_rnn.batch_size,
            shuffle = False
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),lr = hp_rnn.learning_rate)

        print('Train Start...')

        for epoch in range(hp_rnn.num_epochs):
            
            start_time = time.time()

            for idx, data in enumerate(trainLoader, 0):

                # y : audio(8000)
                x, y = data
                x = Variable(x)# x : accel(2000,2)

                y = Variable(y.type(torch.FloatTensor), requires_grad = False)
                y = Flatten_fun(y)
                y = y.double()

                optimizer.zero_grad()
                loss = 0
                hidden = model.init_hidden()

                outputs = self.model.forward(x)
                loss = criterion(outputs,y)

                loss.backward()
                optimizer.step()
                self.lossHistory.append((epoch,idx,loss.data[0]))

            print ('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                   %(epoch+1, hp_cnn.num_epochs,idx+1, hp_cnn.num_data//hp_cnn.batch_size, loss.data[0]))
            print("--- %s seconds for epoch ---" % (time.time() - start_time))
            self.save(self.outPath, 'epoch' + str(epoch))

        self.save(self.outPath, 'final')
        print(self.lossHistory)


    def test(self):
        return 






