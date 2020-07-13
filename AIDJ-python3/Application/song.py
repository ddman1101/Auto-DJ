# Copyright 2017 Len Vande Veire, IDLab, Department of Electronics and Information Systems, Ghent University
# This file is part of the source code for the Auto-DJ research project, published in Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system.", 2018 (submitted)
# Released under AGPLv3 license.

from util import *
import time
from essentia import *
from essentia.standard import *
import numpy as np
from scipy.stats import skew

from BeatTracker import *
# from DownbeatTracker.downbeatTracker import *
import downbeatTrack
# import BeatTracker_Test
from structuralsegmentation import *
from keyestimation import KeyEstimator

import songtransitions
from timestretching import * # Data augmentation

import logging
logger = logging.getLogger('colorlogger')

from sklearn.externals import joblib
theme_pca      = joblib.load('song_theme_pca_model_2.pkl') 
theme_scaler   = joblib.load('song_theme_scaler_2.pkl') 
singing_model  = joblib.load('singingvoice_model.pkl')
singing_scaler = joblib.load('singingvoice_scaler.pkl')

def normalizeAudioGain(audio, rgain, target = -10):
	# Normalize to a level of -16, which is often value of signal here
	factor = 10**((-(target - rgain)/10.0) / 2.0) # Divide by 2 because we want sqrt (amplitude^2 is energy)
	audio *= factor	
	return audio

class Song:
	
	def __init__(self, path_to_file):
		
		self.dir_, self.title = os.path.split(os.path.abspath(path_to_file))
		self.title, self.extension = os.path.splitext(self.title)
		self.dir_annot = self.dir_ + '/' + ANNOT_SUBDIR
		
		if not os.path.isdir(self.dir_annot):
			logger.debug('Creating annotation directory : ' + self.dir_annot)
			os.mkdir(self.dir_annot)
		
		self.audio = None
		self.beats = None
		self.onset_curve = None
		self.tempo = None
		self.downbeats = None
		self.segment_indices = None
		self.segment_types = None
		self.songBeginPadding = 0	# Number of samples to pad the song with, if first segment index < 0
		self.replaygain = None
		self.key = None
		self.scale = None # Major or minor
		self.spectralContrast = None
		
		# Features shared over different components (e.g. beat and downbeat tracking)
		self.fft_phase_1024_512 = None
		self.fft_mag_1024_512 = None
	
	def getSegmentType(self, dbeat):
		''' Get the segment type ('H' or 'L') of the segment the dbeat falls in '''	
		for i in range(len(self.segment_types)-1):
			if self.segment_indices[i] <= dbeat and self.segment_indices[i+1] > dbeat:
				return self.segment_types[i]
		raise Exception('Invalid downbeat ' + str(dbeat) + ', should be between ' + str(self.segment_indices[0]) + ', ' + str(self.segment_indices[-1]))
	
	def hasAnnot(self, prefix):
		return os.path.isfile(pathAnnotationFile(self.dir_, self.title, prefix))
	
	def hasBeatAnnot(self):
		return self.hasAnnot(ANNOT_BEATS_PREFIX)
		
	def hasOnsetCurveAnnot(self):
		return self.hasAnnot(ANNOT_ODF_HFC_PREFIX)
		
	def hasDownbeatAnnot(self):
		return self.hasAnnot(ANNOT_DOWNB_PREFIX)
		
	def hasSegmentationAnnot(self):
		return self.hasAnnot(ANNOT_SEGMENT_PREFIX)
		
	def hasReplayGainAnnot(self):
		return self.title in loadCsvAnnotationFile(self.dir_, ANNOT_GAIN_PREFIX)
		
	def hasKeyAnnot(self):
		return self.title in loadCsvAnnotationFile(self.dir_, ANNOT_KEY_PREFIX)
		
	def hasSingingVoiceAnnot(self):
		return self.hasAnnot(ANNOT_SINGINGVOICE_PREFIX)
		
	def hasSpectralContrastSummaryAnnot(self):
		return self.hasAnnot(ANNOT_SPECTRALCONTRAST_PREFIX)
		
	def hasAllAnnot(self):
		'''
		Check if this file has annotation files
		'''
		return (
			self.hasBeatAnnot() 
			and self.hasOnsetCurveAnnot() 
			and self.hasDownbeatAnnot() 
			and self.hasSegmentationAnnot() 
			and self.hasReplayGainAnnot()
			and self.hasKeyAnnot()
			and self.hasSpectralContrastSummaryAnnot()
			and self.hasSingingVoiceAnnot()
			)
		
	def annotate(self):
		# This doesn't store the annotations and audio in memory yet, this would cost too much memory: writes the annotations to disk and evicts the data from main memory until the audio is loaded for playback
		loader = MonoLoader(filename = os.path.join(self.dir_, self.title + self.extension))
		self.audio = loader()
		
		# Beat and downbeat annotations; have to happen at the same time because of feature sharing (more efficient calculation instead of repeating FFT)		
		if not self.hasBeatAnnot() or not self.hasDownbeatAnnot():
			logger.debug('Annotating beats of ' + self.title)
			btracker = BeatTracker()
			btracker.run(self.audio)
			self.beats = btracker.getBeats()	# Do not keep the annotations in memory, so NOT self.beats = ...
			self.tempo = btracker.getBpm()
			self.phase = btracker.getPhase()
			self.onset_curve = btracker.onset_curve
			self.fft_mag_1024_512 = btracker.fft_mag_1024_512
			self.fft_phase_1024_512 = btracker.fft_phase_1024_512
			writeAnnotFile(self.dir_, self.title, ANNOT_BEATS_PREFIX, self.beats, {'tempo' : self.tempo})
			
			logger.debug('Annotating downbeats of ' + self.title)
#			dbtracker = DownbeatTracker()
			# Reuse FFT results from the beat tracker
			self.downbeats = downbeatTrack.get_downbeats(os.path.join(self.dir_, self.title + self.extension))
			writeAnnotFile(self.dir_, self.title, ANNOT_DOWNB_PREFIX, self.downbeats)
			writeBinaryAnnotFile(self.dir_, self.title, ANNOT_ODF_HFC_PREFIX, self.onset_curve)
		else:
			self.loadAnnotBeat()
			self.loadAnnotDownbeat()
			self.loadAnnotOnsetCurve()
			
		# Segmentation annotations
		if not self.hasSegmentationAnnot():
			logger.debug('Annotating structural segment boundaries of ' + self.title)
			structuralSegmentator = StructuralSegmentator()
			self.segment_indices, self.segment_types = structuralSegmentator.analyse(self)
			writeAnnotFile(self.dir_, self.title, ANNOT_SEGMENT_PREFIX, list(zip(self.segment_indices, self.segment_types)))			
		else:
			self.loadAnnotSegments()
			
		if not self.hasReplayGainAnnot():
			logger.debug('Annotating replay gain of ' + self.title)
			# Dividing by 2 decreases the replay gain (related to energy in db) by 6,... dB
			# Indeed, amplitude / 2 => energy / 4 => 10log(4) = 6,...
			replayGain = ReplayGain()
			rgain = replayGain(self.audio)
			writeCsvAnnotation(self.dir_, ANNOT_GAIN_PREFIX, self.title, rgain)
		
		if not self.hasKeyAnnot():
			logger.debug('Annotating key of ' + self.title)
			#~ keyExtractor = KeyExtractor()
			keyExtractor = KeyEstimator()
			key, scale = keyExtractor(self.audio)
			writeCsvAnnotation(self.dir_, ANNOT_KEY_PREFIX, self.title, key + ':' + scale)
		
		if not self.hasSpectralContrastSummaryAnnot():
			H_number = 0
			for i in range(len(self.segment_types)):
				if self.segment_types[i] == "H":
					H_number = H_number + 1
				else :
					H_number = H_number 
			if H_number == 0 :
				H_tmp = np.random.randint(2,len(self.segment_types)-2)
				self.segment_types[H_tmp] = "H"
            
			segments_H = [i for i in range(len(self.segment_types)) if self.segment_types[i] == 'H']
			# Calculate the spectral contrast features for the audio frames that fall in the H segments
			# Only the H segments are considered as these are most representative of the entire audio
			FRAME_SIZE = 2048		# About 1 beats at 172 BPM and 44100 Hz sample rate
			HOP_SIZE = int(FRAME_SIZE/2)	# About 0.5 beat interval at 172 BPM 
			
			spec = Spectrum(size = FRAME_SIZE)
			w = Windowing(type = 'hann')
			fft = np.fft.fft
			pool = Pool()
			specContrast = SpectralContrast(frameSize=FRAME_SIZE, sampleRate=44100,numberBands=12)
			
			logger.debug('Annotating spectral contrast descriptor of ' + self.title)
						
			for i in segments_H:
				start_sample = int(44100*self.downbeats[self.segment_indices[i]])
				end_sample = int(44100*self.downbeats[self.segment_indices[i+1]])
				for frame in FrameGenerator(self.audio[start_sample:end_sample], frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
					frame_spectrum = spec(w(frame))
					specCtrst, specValley = specContrast(frame_spectrum)
					pool.add('audio.spectralContrast', specCtrst)
					pool.add('audio.spectralValley', specValley)
				
			def calculateDeltas(array):
				D = array[1:] - array[:-1]
				return D	
			specCtrstAvgs = np.average(pool['audio.spectralContrast'], axis=0)
			specValleyAvgs = np.average(pool['audio.spectralValley'], axis=0)
			specCtrstDeltas = np.average(np.abs(calculateDeltas(pool['audio.spectralContrast'])), axis=0)
			specValleyDeltas = np.average(np.abs(calculateDeltas(pool['audio.spectralValley'])), axis=0)
			writeBinaryAnnotFile(self.dir_, self.title, ANNOT_SPECTRALCONTRAST_PREFIX, np.concatenate((specCtrstAvgs,specValleyAvgs, specCtrstDeltas, specValleyDeltas)))
		
		if not self.hasSingingVoiceAnnot():
			
			logger.debug('Annotating voice annotations of ' + self.title)
			
			pool = Pool()
			
			def calculate_sing_features(audio):
								
				features = []
				
				FRAME_SIZE, HOP_SIZE = 2048, 1024
				low_f = 100
				high_f = 7000
				
				w = Windowing(type = 'hann')
				spec = Spectrum(size = FRAME_SIZE)
				mfcc = MFCC(lowFrequencyBound=low_f, highFrequencyBound=high_f)
				spectralContrast = SpectralContrast(lowFrequencyBound=low_f, highFrequencyBound=high_f)
				pool = Pool()
				
				for frame in FrameGenerator(audio, frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
					frame_spectrum = spec(w(frame))
					spec_contrast, spec_valley = spectralContrast(frame_spectrum)
					mfcc_bands, mfcc_coeff = mfcc(frame_spectrum)
					pool.add('spec_contrast', spec_contrast)
					pool.add('spec_valley', spec_valley)
					pool.add('mfcc_coeff', mfcc_coeff)
					
				def add_moment_features(array):
					avg = np.average(array,axis=0)
					std = np.std(array,axis=0)
					skew = scipy.stats.skew(array,axis=0)
					deltas = array[1:,:] - array[:-1,:]
					avg_d = np.average(deltas,axis=0)
					std_d = np.std(deltas,axis=0)
					#~ skew_d = scipy.stats.skew(deltas,axis=0)
				
					features.extend(avg)
					features.extend(std)
					features.extend(skew)
					features.extend(avg_d)
					features.extend(std_d)
					#~ features.extend(skew_d) # Does not contribute a lot
				
				add_moment_features(pool['spec_contrast'])
				add_moment_features(pool['spec_valley'])
				add_moment_features(pool['mfcc_coeff'])
				
				return np.array(features,dtype='single')
			
			features = []
			for dbeat_idx in range(len(self.downbeats)-1):	
				start = int(self.downbeats[dbeat_idx]*44100)
				stop = int(self.downbeats[dbeat_idx+1]*44100)
				if start >= len(self.audio):
					break
				features.append(calculate_sing_features(self.audio[start:stop]))			
			X = np.array(features)
			
			is_singing = np.array(
					singing_model.decision_function(singing_scaler.transform(X))
				, dtype='single')
			writeBinaryAnnotFile(self.dir_, self.title, ANNOT_SINGINGVOICE_PREFIX, is_singing)
		
		# Clean up memory: do not keep annotations or audio in RAM until song is actually used
		self.close()
		
		
	def loadAnnotBeat(self):
		beats_str, res_dict = loadAnnotationFile(self.dir_, self.title, ANNOT_BEATS_PREFIX)
		self.beats = [float(b) for b in beats_str]
		self.tempo = res_dict['tempo']
		
	def loadAnnotOnsetCurve(self):
		self.onset_curve = loadBinaryAnnotFile(self.dir_, self.title, ANNOT_ODF_HFC_PREFIX)
		
	def loadAnnotDownbeat(self):
		downbeats_str, res_dict = loadAnnotationFile(self.dir_, self.title, ANNOT_DOWNB_PREFIX)
		self.downbeats = [float(b) for b in downbeats_str]
		
	def loadAnnotSegments(self):		
		segment_annot_strs, segment_annot_dict  = loadAnnotationFile(self.dir_, self.title, ANNOT_SEGMENT_PREFIX)
		self.segment_indices = []
		self.segment_types = []
		
		for s in segment_annot_strs:
			s1,s2 = s.split(' ')[:2]
			self.segment_indices.append(int(s1))
			self.segment_types.append(s2)
			
		self.loadAnnotSegments_fixNegativeStart()
		
	def loadAnnotSegments_fixNegativeStart(self):
		# Some songs have a negative first segment index because the first measure got cropped a bit
		# The song is therefore extended artificially by introducing silence at the beginning
		if self.segment_indices[0] < 0:
			# Calculate the amount of padding
			beat_length_s = 60.0 / self.tempo
			songBeginPaddingSeconds = (-self.segment_indices[0] * 4 * beat_length_s - self.downbeats[0])
			self.songBeginPadding = int(songBeginPaddingSeconds * 44100 )
			self.downbeats = [dbeat + songBeginPaddingSeconds for dbeat in self.downbeats]
			self.downbeats = [i*4*beat_length_s for i in range(-self.segment_indices[0])] + self.downbeats
			songBeginPaddingSeconds = (-self.segment_indices[0] * 4 * beat_length_s - self.downbeats[0])
			self.songBeginPadding = int(songBeginPaddingSeconds * 44100 )
			self.beats = [beat + songBeginPaddingSeconds for beat in self.beats]
			self.beats = [i*beat_length_s for i in range(int(self.beats[0] / beat_length_s))] + self.beats
			self.onset_curve = np.append(np.zeros((1,int(self.songBeginPadding / 512))), self.onset_curve) # 512 is hop size for OD curve calculation
			offset = self.segment_indices[0] 
			self.segment_indices = [idx - offset for idx in self.segment_indices]
	
	def loadAnnotGain(self):
		self.replaygain = loadCsvAnnotationFile(self.dir_, ANNOT_GAIN_PREFIX)[self.title]
	
	def loadAnnotKey(self):
		self.key, self.scale = loadCsvAnnotationFile(self.dir_, ANNOT_KEY_PREFIX)[self.title].split(':')	
		
	def loadAnnotSpectralContrast(self):
		self.spectral_contrast = loadBinaryAnnotFile(self.dir_, self.title, ANNOT_SPECTRALCONTRAST_PREFIX).reshape((-1))
	
	def loadAnnotSingingVoice(self):
		self.singing_voice = loadBinaryAnnotFile(self.dir_, self.title, ANNOT_SINGINGVOICE_PREFIX).reshape((-1))	
	
	def calculateSongThemeDescriptor(self):
		features = self.spectral_contrast
		self.song_theme_descriptor = theme_pca.transform(theme_scaler.transform(
			features.astype('single').reshape((1,-1))
			)).astype('single')
	
	# Open the audio file and read the annotations
	def open(self):
		self.loadAnnotBeat()
		self.loadAnnotOnsetCurve()
		self.loadAnnotDownbeat()
		self.loadAnnotSegments()	
		self.loadAnnotKey()
		self.loadAnnotGain()
		self.loadAnnotSpectralContrast()
		self.loadAnnotSingingVoice()
		self.calculateSongThemeDescriptor()
			
	def openAudio(self):
		#~ logger.debug('Opening audio ' + str(self.title))
		loader = MonoLoader(filename = os.path.join(self.dir_, self.title + self.extension))
		time0 = time.time()
		audio = loader().astype('single')
		time1 = time.time()
		#~ logger.debug('Time waiting for audio loading: ' + str(time1-time0))
		self.audio = normalizeAudioGain(audio, self.replaygain)
		if self.songBeginPadding > 0:
			self.audio = np.append(np.zeros((1,self.songBeginPadding),dtype='single'), self.audio)
	
	def closeAudio(self):
		# Garbage collector will take care of this later on
		self.audio = None
		
	# Close the audio file and reset all buffers to None
	def close(self):
		self.audio = None
		self.beats = None
		self.onset_curve = None
		self.tempo = None
		self.downbeats = None
		self.segment_indices = None
		self.segment_types = None
		self.replaygain = None
		self.key = None
		self.scale = None # Major or minor
		self.spectralContrast = None
		self.fft_phase_1024_512 = None
		self.fft_mag_1024_512 = None
		
	def getOnsetCurveFragment(self, start_beat_idx, stop_beat_idx, target_tempo = 175):
		# Cut out a section of the onset detection curve with beat granularity
		# Stretch the onset curve to the target BPM (175 by default) to ensure that
		# curves of different songs are comparable
		
		# Parameters of the onset detection function calculation
		HOP_SIZE = 512
		SAMPLE_RATE = 44100
		start_frame = int(int(SAMPLE_RATE * self.beats[start_beat_idx]) / HOP_SIZE)
		stop_frame = int(int(SAMPLE_RATE * self.beats[stop_beat_idx]) / HOP_SIZE)
		return self.onset_curve[start_frame:stop_frame]
		
	def markedForAnnotation(self):
		# Songs can be marked for manual annotation fixing. This is stored in this annotation file
		return self.title in loadCsvAnnotationFile(self.dir_, ANNOT_MARKED_PREFIX)
		
	def markForAnnotation(self):
		writeCsvAnnotation(self.dir_, ANNOT_MARKED_PREFIX, self.title, 1)
		
	def unmarkForAnnotation(self):
		deleteCsvAnnotation(self.dir_, ANNOT_MARKED_PREFIX, self.title)
		
