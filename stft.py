#STFT
#wav to spectrogram
#spectrogram to wav
import argparse
import os
import numpy as np
import librosa
import hparams as hp
from hparams import Default as hp_default
import pickle

def normalizeSpectrogram(spectro, mean = None, std = None):
    
	if mean == None or std == None:
		mean = np.mean(spectro[:])
		std = np.std(spectro[:])
		normalized = (spectro - mean) / (np.sqrt(2.0) * std + 1e-13)
	return normalized

#wav to spectrogram
#def transform(filePath):
def wav2spctro(filePath):
	result = list()
	#load audio
	audio, rate = librosa.load(filePath, mono=True, sr = hp_default.sr)
	print(audio)
	print('audio :', audio.shape[0])
	print('rate :', rate)
	spectro = librosa.stft(audio,
		n_fft = hp_default.n_fft,
		hop_length = hp_default.hop_length,
		win_length = hp_default.win_length
		)
	print ('spectro')
	print(spectro.shape[0])
	print(spectro.shape[1])
	#Normaliztio

	#paddding
	xsize = int(1+hp_default.sr*hp_default.timeLength/hp_default.hop_length)
	print (xsize)
	if spectro.shape[1] < xsize:
		print('add padding')
		spectro = np.lib.pad(spectro, ((0, 0), (0, xsize - spectro.shape[1])), 'constant', constant_values = 1e-13)
		spectro_ = spectro.T
	if spectro.shape[1] > xsize:
		spectro_ = spectro.T[:xsize][:]
		print('cut')
	if spectro_.shape[0] == xsize:
		result.append(np.abs(spectro_))
		print(6)
	return result #(t 101, 1+ n_ftt/2 257)
#wav to spectrogram
#def transform(filePath):

def wav2spctro2(filePath):
	outPath = "input_spectro_0213"
	result = list()
	#load audio
	audio, rate = librosa.load("wav_0213\wood_hit.wav", mono=True, sr = hp_default.sr)
	print(audio)
	print('audio :', audio.shape[0])
	print('rate :', rate)
	temp = 12800 # 40/50 * 16000
	for i in range(1,166):
		spectro = librosa.stft(audio[temp:temp+8000],
			n_fft = hp_default.n_fft,
			hop_length = hp_default.hop_length,
			win_length = hp_default.win_length
			)
		print ('spectro')
		print(spectro.shape[0])
		print(spectro.shape[1])
		#Normaliztio

		#paddding
		xsize = int(1+hp_default.sr*hp_default.timeLength/hp_default.hop_length)
		print (xsize)
		if spectro.shape[1] < xsize:
			print('add padding')
			spectro = np.lib.pad(spectro, ((0, 0), (0, xsize - spectro.shape[1])), 'constant', constant_values = 1e-13)
			spectro_ = spectro.T
		if spectro.shape[1] > xsize:
			spectro_ = spectro.T[:xsize][:]
			print('cut')
		if (spectro.shape[1] == xsize):
			spectro_ = spectro.T

		if spectro_.shape[0] == xsize:
			result.append(np.abs(spectro_))
			print(6)

		outFile = 'spectro_wood_' + str(i) + '.pickle'
		with open(os.path.join(outPath,outFile),'wb') as fs:
			pickle.dump(result,fs)

		temp += 16000
		result = list()
	return result #(t 101, 1+ n_ftt/2 257)
'''
(1+n_fft/2, 1+sr*timeLength/hop_length)
(257,101)-> (101,257)
'''

#spectrogram to wav
def griffinLim(spectro, iterN = 50):
	# reference : https://github.com/andabi/deep-voice-conversion/blob/master/tools/audio_utils.py
	phase = np.pi * np.random.rand(*spectro.shape)
	print("griffinLim")
	print(spectro.shape)
	spec = spectro * np.exp(1.0j * phase)
	for i in range(iterN):
		audio = librosa.istft(spec, hop_length = hp_default.hop_length, win_length = hp_default.win_length)
		if i < iterN - 1:
			spec = librosa.stft(audio, n_fft = hp_default.n_fft, hop_length = hp_default.hop_length, win_length = hp_default.win_length)
			_, phase = librosa.magphase(spec)
			spec = spectro * np.exp(1.0j * np.angle(phase))
	return audio

'''
if mode is transform, it transform wav to spectrogram
if inverse, spectrogram to wav
'''

def transform(inPath, outPath, mode = 'convert'):
	if mode == 'transform':
		if not os.path.exists(outPath):
			os.makedirs(outPath)
		for path, dirs, files in os.walk(inPath):
			for f in files:
				if os.path.splitext(f)[-1] == '.wav':
#					try:
					print(os.path.join(path,f))
					spectro = wav2spctro(os.path.join(path,f))
					print('a')
#					   for idx, spectro in enumerate(spectroList):
					outFile = 'spectro_' + '_' + os.path.splitext(f)[0] + '.pickle'
					print('b')
					with open(os.path.join(outPath,outFile),'wb') as fs:
						pickle.dump(spectro,fs)
						print('c')
#					except:
#						print('error')
#						continue


	elif mode =='inverse':
		if not os.path.exists(outPath):
			os.makedirs(outPath)

		for path, dirs, files in os.walk(inPath):
			for f in files:
				if os.path.splitext(f)[-1] == '.pickle':

					with open(os.path.join(path,f), 'rb') as fs:
						spectro = pickle.load(fs)
						print(spectro)
						spectro = spectro[0]
						print(spectro)
						x= np.asarray(spectro)
						print(len(spectro))
						print(len(spectro[0]))
						print(x.shape)
						audio = griffinLim(x.T)
						print(0)
						filename='converted_'+os.path.basename(path)+os.path.splitext(f)[0]+'.wav'
						print(1)
						librosa.output.write_wav(os.path.join(outPath,filename),audio,sr=hp_default.sr)
	
#I don't know why  
	elif mode =='inverse2':
		if not os.path.exists(outPath):
			os.makedirs(outPath)

		for path, dirs, files in os.walk(inPath):
			for f in files:
				if os.path.splitext(f)[-1] == '.pickle':
					with open(os.path.join(path,f), 'rb') as fs:
						spectro = pickle.load(fs)

						print(type(spectro))
						spectro = np.asarray(spectro)
						spectro.reshape((257,101))
						print(type(spectro))
						print(spectro.shape)
						#for idx2,each_line in enumerate(spectro) :
						#	each_line = [float(i) for i in each_line]
						#	spectro[idx2] = each_line
						audio = griffinLim(spectro.T)
						filename='converted_'+os.path.basename(path)+os.path.splitext(f)[0]+'.wav'
						librosa.output.write_wav(os.path.join(outPath,filename),audio,sr=hp_default.sr)

#test
def test(spectro, outPath, f):
	audio = griffinLim(spectro.T)
	filename='converted_'+os.path.basename(outPath)+os.path.splitext(f)[0]+'.wav'
	librosa.output.write_wav(os.path.join(outPath,filename),audio,sr=hp_default.sr)

#wav to spectrogram
#transform('input_audio_0213','input_spectro_0213','transform')
#spectrogram to wav file
#main('output_vocal','converted_wav','inverse')

#transform('output_spectro_test','converted_wav_test','inverse2')


#transform('input_spectro_0213','testtesttest','inverse')


#wav2spctro2("a")