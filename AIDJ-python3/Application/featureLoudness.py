import numpy as np
from essentia import *
from essentia.standard import Windowing, Loudness
from sklearn import preprocessing

def feature_allframes(audio, beats, frame_indexer = None):
	
	# Initialise the algorithms
	w = Windowing(type = 'hann')
	loudness = Loudness()
	
	if frame_indexer is None:
		frame_indexer = list(range(1,len(beats) - 1)) # Exclude first frame, because it has no predecessor to calculate difference with
		
	# 1 loudness value by default
	loudness_values = np.zeros((len(beats), 1))
	# 1 difference value between loudness value cur and cur-1
	# 1 difference value between loudness value cur and cur-4
	# 1 difference value between differences above
	loudness_differences = np.zeros((len(beats), 9))
	
	# Step 1: Calculate framewise for all output frames
	# Calculate this for all frames where this frame, or its successor, is in the frame_indexer
	for i in [i for i in range(len(beats)) if (i in frame_indexer) or (i+1 in frame_indexer) 
		or (i-1 in frame_indexer) or (i-2 in frame_indexer) or (i-3 in frame_indexer) or (i-4 in frame_indexer)
		or (i-5 in frame_indexer) or (i-6 in frame_indexer) or (i-7 in frame_indexer) or (i-8 in frame_indexer)]:
		
		SAMPLE_RATE = 44100
		start_sample = int(beats[i] * SAMPLE_RATE)
		end_sample = int(beats[i+1] * SAMPLE_RATE) 
		#print start_sample, end_sample
		frame = audio[start_sample : end_sample if (start_sample - end_sample) % 2 == 0 else end_sample - 1]
		loudness_values[i] = loudness(w(frame))
		
	# Step 2: Calculate the cosine distance between the MFCC values
	for i in frame_indexer:
		loudness_differences[i][0] = (loudness_values[i] - loudness_values[i-1])
		loudness_differences[i][1] = (loudness_values[i+1] - loudness_values[i])
		loudness_differences[i][2] = (loudness_values[i+2] - loudness_values[i])
		loudness_differences[i][3] = (loudness_values[i+3] - loudness_values[i])
		loudness_differences[i][4] = (loudness_values[i+4] - loudness_values[i])
		loudness_differences[i][5] = (loudness_values[i+5] - loudness_values[i])
		loudness_differences[i][6] = (loudness_values[i+6] - loudness_values[i])
		loudness_differences[i][7] = (loudness_values[i+7] - loudness_values[i])
		loudness_differences[i][8] = (loudness_values[i-1] - loudness_values[i+1])
		
	# Include the raw values as absolute features
	result = loudness_differences[frame_indexer]
	
	#~ print np.shape(result), np.shape(loudness_values), np.shape(loudness_differences)
	return preprocessing.scale(result)
