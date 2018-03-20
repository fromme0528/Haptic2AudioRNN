import librosa

def AudioRead(inPath):
	audio, rate = librosa.load(inPath) 
	print (audio, rate)

#AudioRead("./dataset/audio_split/audio_plastic_0.wav")