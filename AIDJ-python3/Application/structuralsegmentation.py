# Copyright 2017 Len Vande Veire, IDLab, Department of Electronics and Information Systems, Ghent University
# This file is part of the source code for the Auto-DJ research project, published in Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system.", 2018 (submitted)
# Released under AGPLv3 license.

from essentia import *
from essentia.standard import *
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.signal

essentia.log_active = False

import logging
logger = logging.getLogger('colorlogger')

def calculateCheckerboardCorrelation(matrix, N):
	
	M = min(matrix.shape[0], matrix.shape[1])
	result = np.zeros(M)
	
	u1 = scipy.signal.gaussian(2*N, std=N/2.0).reshape((2*N,1))
	u2 = scipy.signal.gaussian(2*N, std=N/2.0).reshape((2*N,1))
	U = np.dot(u1,np.transpose(u2))
	U[:N,N:] *= -1
	U[N:,:N] *= -1
	
	matrix_padded = np.pad(matrix, N, mode='edge')
	
	for index in range(N, N+M):
		submatrix = matrix_padded[index-N:index+N, index-N:index+N]
		result[index-N] = np.sum(submatrix * U)
	return result

def adaptive_mean(x, N):
	return np.convolve(x, [1.0]*int(N), mode='same')/N

class StructuralSegmentator:
	
	def analyse(self, song): #(audio_in, downbeats, tempo):
		
		audio_in = song.audio
		downbeats = song.downbeats
		tempo = song.tempo
		
		# Initialize algorithm objects
		pool = Pool()
		w = Windowing(type = 'hann')
		spectrum = Spectrum()
		mfcc = MFCC()
		
		# Cut the audio so it starts at a (guessed) downbeat
		first_downbeat_sample = int(44100 * downbeats[0])
		audio = audio_in[ first_downbeat_sample : ]

		# -------------- SELF-SIMILARITY MATRICES ----------------------
		
		# MFCC self-similarity matrix and novelty curve
		# TODO: These features are also calculated for beat tracking -> can be optimized
		FRAME_SIZE = int(44100 * (60.0 / tempo) / 2)
		HOP_SIZE = int(FRAME_SIZE / 2)
		for frame in FrameGenerator(audio, frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
			mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame[:FRAME_SIZE-(FRAME_SIZE % 2)])))
			pool.add('lowlevel.mfcc', mfcc_coeffs)
			pool.add('lowlevel.mfcc_bands', mfcc_bands)

		selfsim_mfcc = cosine_similarity(np.array(pool['lowlevel.mfcc']), np.array(pool['lowlevel.mfcc']))
		selfsim_mfcc -= np.average(selfsim_mfcc)
		selfsim_mfcc *= (1.0 / np.max(selfsim_mfcc))

		# RMS self-similarity matrix and novelty curve
		for frame in FrameGenerator(audio, frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
			pool.add('lowlevel.rms', np.average(frame**2))

		selfsim_rms = pairwise_distances(pool['lowlevel.rms'].reshape(-1, 1))
		selfsim_rms -= np.average(selfsim_rms)
		selfsim_rms *= (1.0 / np.max(selfsim_rms))
		
		# -------------- NOVELTY CURVES --------------------------------
		novelty_mfcc = calculateCheckerboardCorrelation(selfsim_mfcc, N = 32)
		novelty_mfcc *= 1.0/np.max(novelty_mfcc)
		
		novelty_rms = np.abs(calculateCheckerboardCorrelation(selfsim_rms, N = 32))
		novelty_rms *= 1.0/np.max(np.abs(novelty_rms))

		# Product-signal accentuates the peaks that are present in both signals at the same time
		novelty_product = novelty_rms * novelty_mfcc
		novelty_product = [i if i > 0 else 0 for i in novelty_product]
		novelty_product = np.sqrt(novelty_product)

		# -------------- PEAK PICKING ----------------------------------
		# Find the most dominant peaks in the product signal
		peaks_absmax_i = np.argmax(novelty_product)				
		peaks_absmax = novelty_product[peaks_absmax_i]			# The absolute maximum its index
		threshold = peaks_absmax * 0.05							# Detect peaks at least 1/20th as high
		# Detect other peaks
		peakDetection = PeakDetection(interpolate=False, maxPeaks=100, orderBy='amplitude', range=len(novelty_product), maxPosition=len(novelty_product), threshold=threshold)
		peaks_pos, peaks_ampl = peakDetection(novelty_product.astype('single'))
		peaks_ampl = peaks_ampl[np.argsort(peaks_pos)]
		peaks_pos = peaks_pos[np.argsort(peaks_pos)]

		# Filter the peaks
		# Shift the peaks that are in window (-delta * downbeatLength, delta * downbeatLength) 
		# to the peak in the center of that interval.
		# Peaks that are not within that interval are removed as false positives
		peaks_pos_modified, peaks_ampl_modified = [], []
		peaks_pos_dbindex = []

		peak_idx = 0
		peak_cur_s = (HOP_SIZE / 44100.0) * peaks_pos[peak_idx]
		num_filtered_out = 0

		downbeat_len_s = 4 * 60.0 / tempo
		delta = 0.4
		for dbindex, downbeat in zip(list(range(len(downbeats))), np.array(downbeats) - downbeats[0]):
			# Skip the peaks prior to the acceptance interval
			while peak_cur_s < downbeat - delta * downbeat_len_s and peak_idx < len(peaks_pos):
				num_filtered_out += 1
				peak_idx += 1
				if peak_idx != len(peaks_pos):
					peak_cur_s = (HOP_SIZE / 44100.0) * peaks_pos[peak_idx]
			if peak_idx == len(peaks_pos):
				break
			# Adjust the peaks within the acceptance interval
			while peak_cur_s < downbeat + delta * downbeat_len_s and peak_idx < len(peaks_pos):
				peak_newpos = int(downbeat * 44100.0 / HOP_SIZE)	# seconds to frames
				peaks_pos_modified.append(peak_newpos)
				peaks_ampl_modified.append(peaks_ampl[peak_idx])
				peaks_pos_dbindex.append(dbindex)
				
				peak_idx += 1
				if peak_idx != len(peaks_pos):
					peak_cur_s = (HOP_SIZE / 44100.0) * peaks_pos[peak_idx]
			if peak_idx == len(peaks_pos):
				break
				
		peaks_pos_modified = np.array(peaks_pos_modified)		# The modified positions of the peaks
		peaks_ampl_modified = np.array(peaks_ampl_modified)		# The amplitudes of the peaks
		peaks_pos_dbindex = np.array(peaks_pos_dbindex)			# The downbeat indices of the peaks

		# Determine the most dominant peaks and see if they lie at a multiple of 8 downbeats from each 
		# Assumption 1: high peaks are important; assumption 2: they should lie at multiples of 8 downbeats (phrase) from each other
		NUM_HIGHEST_PEAKS = 20
		highest_peaks_db_indices = (peaks_pos_dbindex[np.argsort(peaks_ampl_modified)])[-NUM_HIGHEST_PEAKS:]
		highest_peaks_amplitudes = (peaks_ampl_modified[np.argsort(peaks_ampl_modified)])[-NUM_HIGHEST_PEAKS:]
		distances = []			# Number of total peaks at multiple of 4
		distances_high = [] 	# Number of high peaks at multiple of 4 for each of the highest 
		distances8 = []			# Number of total peaks at multiple of 8
		distances8_high = [] 	# Number of high peaks at multiple of 8 for each of the highest 

		# Look at which 8-downbeat cyclic offset the segments lie
		for i in range(8):
			# Count at which offset of 8 downbeats the peaks are.
			# However, if a peak is one or two downbeats before a high peak, then that's probably an accentuation of that change
			distances8.append(sum( [h for p,h in zip(highest_peaks_db_indices, highest_peaks_amplitudes) if (p - i) % 8 == 0 and (p+1 not in highest_peaks_db_indices or max(highest_peaks_amplitudes[highest_peaks_db_indices==p+1]) < 0.75*h) and (p+2 not in highest_peaks_db_indices or max(highest_peaks_amplitudes[highest_peaks_db_indices==p+2]) < 0.75*h)]))
			distances8_high.append(len( [p for p in highest_peaks_db_indices if (p - i) % 8 == 0] ))
			
		# For the positions where the highest downbeats are detected, determine which one has the most alignments
		# Discard the peaks where no highest peak has been detected
		most_likely_8db_index = np.argmax(distances8 * np.array(distances8_high).astype(float))

		# ----------------- QUEUE POINT SELECTION ----------------------
		# Always have the phrase-aligned start of the song as a segmentation boundary
		last_downbeat = len(downbeats) - 1

		# Always have the start of the song as a segmentation boundary
		# This can be negative, and will be fixed with padding when opening the song
		segment_indices = [most_likely_8db_index if most_likely_8db_index <= 4 else most_likely_8db_index - 8] 	
		# Also have the end of the song as a segmentation boundary
		last_boundary = last_downbeat - ((last_downbeat - most_likely_8db_index) % 8)
		segment_indices.extend([last_boundary])
		# Also have all important segmentation boundaries that were detected earlier
		segment_indices.extend([db for db in highest_peaks_db_indices if (db - most_likely_8db_index) % 8 == 0]) 	# Also have all important segments in there 
		# Important events just before or after a phrase-aligned downbeat could indicate important events too
		segment_indices.extend([db+1 for db in highest_peaks_db_indices if (db + 1 - most_likely_8db_index) % 8 == 0])
		segment_indices.extend([db+2 for db in highest_peaks_db_indices if (db + 2 - most_likely_8db_index) % 8 == 0])
		segment_indices = np.unique(sorted(segment_indices))

		# Determine the type of segment
		adaptive_mean_rms = adaptive_mean(pool['lowlevel.rms'], 64) # Mean of rms in window of [-4 dbeats, + 4 dbeats]
		mean_rms = np.mean(adaptive_mean_rms)
		adaptive_mean_odf = adaptive_mean(song.onset_curve, int((44100*60/tempo)/512) * 4) # -4 dbeats, +4 dbeats
		mean_odf = np.mean(adaptive_mean_odf)
		
		#~ import matplotlib.pyplot as plt
		#~ plt.plot(np.linspace(0.0,1.0,adaptive_mean_rms.size), adaptive_mean_rms / max(adaptive_mean_rms),c='r')
		#~ plt.plot(np.linspace(0.0,1.0,adaptive_mean_odf.size), adaptive_mean_odf / max(adaptive_mean_odf),c='b')
		#~ plt.plot(np.linspace(0.0,1.0,song.onset_curve.size), song.onset_curve / (2*max(song.onset_curve)),c='grey')
		#~ plt.axhline(mean_rms / max(adaptive_mean_rms),c='r')
		#~ plt.axhline(mean_odf / max(adaptive_mean_odf),c='b')
		#~ plt.show()
		
		segment_types = [] 
		
		def getSegmentType(dbindex):
			
			if dbindex >= last_boundary:
				return 'L'
				
			after_index = int(int((dbindex + 4) * 4 * 60.0/tempo * 44100.0)/HOP_SIZE)	# 2 downbeats after the fade
			rms_after = adaptive_mean_rms[after_index] / mean_rms
			
			after_index = int(int((dbindex + 4) * 4 * 60.0/tempo * 44100.0)/512)	# 32 + downbeat index (running average range is [-32,32])
			odf_after = adaptive_mean_odf[after_index] / mean_odf
						
			return 'H' if rms_after >= 1.0 and odf_after >= 1.0 else 'L'			
		
		for segment in segment_indices:
			segment_types.append(getSegmentType(segment))
			
		# Add more segments in between existing ones if the distance is too small
		additional_segment_indices = []
		additional_segment_types = []
		for i in range(len(segment_indices) - 1):
			if segment_indices[i+1] - segment_indices[i] >= 32:				# Segments are too far apart, so add some new ones
				previous_type = segment_types[i]
				for offset in range(16,segment_indices[i+1] - segment_indices[i],16): 	# 16, 32, 48, ..., distance - 16
					if getSegmentType(segment_indices[i] + offset) != previous_type:	# This offset introduces a new type of segment
						additional_segment_indices.append(segment_indices[i] + offset)
						previous_type = 'H' if previous_type == 'L' else 'L'
						additional_segment_types.append(previous_type)
						
		segment_indices = np.append(segment_indices, additional_segment_indices)
		segment_types = np.append(segment_types, additional_segment_types)
		permutation = np.argsort(segment_indices)
		segment_indices = segment_indices[permutation].astype('int')
		segment_types = segment_types[permutation]
		
		return (segment_indices, segment_types)
