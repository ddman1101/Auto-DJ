import numpy as np
from essentia import *
from essentia.standard import Spectrum, Windowing, MelBands, FrameGenerator, Spectrum
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

NUMBER_BANDS = 12
NUMBER_COEFF = 5

def feature_allframes(audio, beats, frame_indexer = None):
	
	# Initialise the algorithms
	w = Windowing(type = 'hann')
	spectrum = Spectrum() 		# FFT would return complex FFT, we only want magnitude
	melbands = MelBands(numberBands = NUMBER_BANDS)
	pool = Pool()
	
	if frame_indexer is None:
		frame_indexer = list(range(4,len(beats) - 1)) # Exclude first frame, because it has no predecessor to calculate difference with
		
	# 13 MFCC coefficients
	# 40 Mel band energies
	mfcc_bands = np.zeros((len(beats), NUMBER_BANDS))
	# 1 cosine distance value between every mfcc feature vector
	# 13 differences between MFCC coefficient of this frame and previous frame
	# 13 differences between MFCC coefficient of this frame and frame - 4	
	# 13 differences between the differences above	
	# Idem for mel band energies
	mfcc_bands_diff = np.zeros((len(beats), NUMBER_BANDS * 4))
	
	# Step 1: Calculate framewise for all output frames
	# Calculate this for all frames where this frame, or its successor, is in the frame_indexer
	for i in [i for i in range(len(beats)) if (i in frame_indexer) or (i+1 in frame_indexer) 
		or (i-1 in frame_indexer) or (i-2 in frame_indexer) or (i-3 in frame_indexer)]:
		SAMPLE_RATE = 44100
		start_sample = int(beats[i] * SAMPLE_RATE)
		end_sample = int(beats[i+1] * SAMPLE_RATE) 
		frame = audio[start_sample : end_sample if (start_sample - end_sample) % 2 == 0 else end_sample - 1]
		bands = melbands(spectrum(w(frame)))
		mfcc_bands[i] = bands
	
	# Step 2: Calculate the cosine distance between the MFCC values
	for i in frame_indexer:
		# The norm of difference is usually very high around downbeat, because of melodic changes there!
		mfcc_bands_diff[i][0*NUMBER_BANDS : 1*NUMBER_BANDS] = mfcc_bands[i+1] - mfcc_bands[i]
		mfcc_bands_diff[i][1*NUMBER_BANDS : 2*NUMBER_BANDS] = mfcc_bands[i+2] - mfcc_bands[i]
		mfcc_bands_diff[i][2*NUMBER_BANDS : 3*NUMBER_BANDS] = mfcc_bands[i+3] - mfcc_bands[i]
		mfcc_bands_diff[i][3*NUMBER_BANDS : 4*NUMBER_BANDS] = mfcc_bands[i] - mfcc_bands[i-1]
			
	result = mfcc_bands_diff[frame_indexer]
	return preprocessing.scale(result)
