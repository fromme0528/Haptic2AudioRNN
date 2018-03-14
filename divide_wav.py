# Devide wav file 
# 44100Hz

# D\divide_wav.py

import os
import librosa
import hparams as hp
from hparams import Default as hp_default


def divide():

	#load audio#hp_default.sr
	audio, rate = librosa.load("wav_0213\water_hit_volumeUp.wav", mono=True, sr = hp_default.sr)
	temp = 4800
	for i in range(1,146):
		librosa.output.write_wav(os.path.join("input_audio_0213","sample_audio_"+str(i)+".wav"),audio[int(temp):int(temp+8000)],sr=hp_default.sr)
		temp += 16000
audio, rate = librosa.load("input_audio_0213\sample_audio_1.wav")
print(audio)
print(rate)
#divide()