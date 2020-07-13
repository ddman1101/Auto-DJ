# Copyright 2017 Len Vande Veire, IDLab, Department of Electronics and Information Systems, Ghent University
# This file is part of the source code for the Auto-DJ research project, published in Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system.", 2018 (submitted)
# Released under AGPLv3 license.

import numpy as np
from scipy import signal, interpolate
import librosa.effects as effects
import librosa.decompose as decompose
import librosa.core as core
import librosa.util

def crossfade(audio1, audio2, length = None):
	'''Crossfade two audio clips, using linear fading by default.
	Assumes MONO input'''
	if length is None:
		length = min(audio1.size, audio2.size)
	profile = ((np.arange(0.0, length)) / length)
	output = (audio1[:length] * profile[::-1]) + (audio2[:length] * profile)
	return output[:length]
	
def time_stretch_hpss(audio, f):
	
	if f == 1.0:
		return audio
	
	stft = core.stft(audio)
	
	# Perform HPSS
	stft_harm, stft_perc = decompose.hpss(stft,kernel_size=31) # original kernel size 31
	
	# OLA the percussive part
	y_perc = librosa.util.fix_length(core.istft(stft_perc, dtype=audio.dtype), len(audio))
	y_perc = time_stretch_sola(y_perc,f)
	
	#~ # Phase-vocode the harmonic part
	#~ stft_stretch = core.phase_vocoder(stft_harm, 1.0/f)
	#~ # Inverse STFT of harmonic
	#~ y_harm = librosa.util.fix_length(core.istft(stft_stretch, dtype=y_perc.dtype), len(y_perc))
	y_harm = librosa.util.fix_length(core.istft(stft_harm, dtype=audio.dtype), len(audio))
	y_harm = librosa.util.fix_length(time_stretch_sola(core.istft(stft_harm, dtype=audio.dtype), f, wsola = True), len(y_perc))
	
	# Add them together
	return y_harm + y_perc

def time_stretch_sola(audio, f, sample_rate = 44100, wsola = False):
	# Assumes mono 44100 kHz audio (audio.size)
	
	if f == 1.0:
		return audio
	
	# Initialise time offsets and window lengths
	frame_len_1 = 4410 if wsola else 4410/8			# Length of a fragment, including overlap at one side; about 100 ms; shouldn't be longer because length of a 16th note is about this period at 175 BPM: otherwise it won't be possible to copy without doubling transients 
	frame_len_1 = int(frame_len_1)	
	overlap_len = frame_len_1/8						# About 8 ms
	overlap_len = int(overlap_len)
	frame_len_2 = frame_len_1 + overlap_len			# Length of a fragment, including overlap at both sides
	frame_len_0 = frame_len_1 - overlap_len			# Length of a fragment, excluding overlaps (unmixed part)
	next_frame_offset_f =  frame_len_1 / f			# keep as a float to prevent rounding errors
	next_frame_offset = int(next_frame_offset_f) 	# keep as a float to prevent rounding errors
	seek_win_len_half = frame_len_1/16				# window total ~ 21,666 ms
	
	def find_matching_frame(frame, theor_center):
		'''
		Find a frame in the neighbourhood of theor_center that maximizes the autocorrelation with the given frame as much as possible.
			
			:returns The start index in the given audio array of the most matching frame. Points to beginning of STABLE part. 
		'''
		# minus overlap_len because theor_start_frame is at the beginning of the constant part, but you have to convolve with possible intro overlap parts
		# minus len(frame) and not overlap_len to avoid errors when frame is at the very end of the input audio, and is not a full overlap part anymore
		cur_win_min = int(theor_center - seek_win_len_half)
		cur_win_max = int(theor_center + seek_win_len_half)
		correlation = signal.fftconvolve(audio[cur_win_min:cur_win_max+len(frame)], frame[::-1], mode='valid') # Faster than np.correlate! cf http://scipy-cookbook.readthedocs.io/items/ApplyFIRFilter.html
		optimum = np.argmax(correlation[:int(2*seek_win_len_half)])
		
		return theor_center  + (optimum - seek_win_len_half)

	# --------------Algorithm------------------
	# Initialise output buffer
	num_samples_out = int(f * audio.size)
	output = np.zeros(num_samples_out)		# f: stretch factor of audio!; prealloc

	num_frames_out = num_samples_out / frame_len_1
	num_frames_out = int(num_frames_out)
	in_ptr_th_f = 0.0
	in_ptr = 0
	isLastFrame = False

	# frame_index is aligned at the beginning of the constant part of the next frame that will be written in output
	for out_ptr in range(0, num_frames_out * frame_len_1, frame_len_1):
		
		# Write the constant part of the frame
		frame_to_copy = audio[int(in_ptr) : int(in_ptr + frame_len_0)]
		output[out_ptr : int(out_ptr + len(frame_to_copy))]  = frame_to_copy # intermediate step to prevent mismatch between out : out+len and in:in+len' 		
		
		# Check if it is still useful to look for a next frame
		# This is not the case when the part that is overlapped with next frame is not complete (or even completely missing) 
		if (in_ptr + frame_len_1 > audio.size):
			frame_to_copy = audio[int(in_ptr + frame_len_0) : int(in_ptr + frame_len_1)]
			output[out_ptr + frame_len_0 : out_ptr + frame_len_0 + len(frame_to_copy)] = frame_to_copy
			return output
		
		# Look for the next frame that matches best when overlapped
		# Method 2: pure WSOLA: match the next frame in the INPUT audio as closely as possible, since this frame is the natural follower of the original
		frame_to_match = audio[int(in_ptr + frame_len_0) : int(in_ptr + frame_len_0 + frame_len_1)]
		if wsola:
			match_ptr = find_matching_frame(frame_to_match, in_ptr_th_f + next_frame_offset_f - overlap_len)
		else:
			match_ptr = int(in_ptr_th_f + next_frame_offset_f) - overlap_len
			
		frame1_overlap = audio[int(in_ptr + frame_len_0) : int(in_ptr + frame_len_1)]
		frame2_overlap = audio[int(match_ptr) : int(match_ptr + overlap_len)]
		
		# Mix the overlap parts of the frames
		temp = crossfade(frame1_overlap, frame2_overlap)
		output[out_ptr + frame_len_0 : out_ptr + frame_len_0 + len(temp)] = temp
		
		# Increase the input pointers
		in_ptr = match_ptr + overlap_len
		in_ptr_th_f += next_frame_offset_f
		
	return np.array(output).astype('single')
	
def time_stretch_and_pitch_shift(audio, f, semitones=0):
	# Stretch the audio by factor f in speed and perform a pitch shift of N semitones
	semitone_factor = np.power(2.0, semitones/12.0)
	
	#~ audio = time_stretch_sola(audio, f*semitone_factor)
	audio = time_stretch_hpss(audio, f*semitone_factor)
	
	if semitones != 0:
		x = list(range(audio.size))
		x_new = np.linspace(0,audio.size-1,int(audio.size / semitone_factor))
		f = interpolate.interp1d(x, audio, kind='quadratic')
		audio = f(x_new)
	return audio
	
if __name__ == '__main__':
	
	import song
	import sys
	
	s = song.Song(sys.argv[1])
	s.open()
	s.openAudio()
	
	audio = time_stretch_and_pitch_shift(s.audio, s.tempo/175.0, semitones=2)
	
	#~ from librosa.effects import hpss
	#~ audio, audio2 = hpss(s.audio)
	
	from essentia import *
	from essentia.standard import *
	writer = MonoWriter(filename='blub.wav')
	writer(audio.astype('single'))
	
