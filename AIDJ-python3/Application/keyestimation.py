# Copyright 2017 Len Vande Veire, IDLab, Department of Electronics and Information Systems, Ghent University
# This file is part of the source code for the Auto-DJ research project, published in Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system.", 2018 (submitted)
# Released under AGPLv3 license.

from essentia import *
from essentia.standard import *

import numpy as np

'''
	Simple class performing key extraction on song audio
'''

class KeyEstimator:
	
	def __init__(self):
		pass
		
	def __call__(self, audio):
		FRAME_SIZE = 2048		# About 1 beats at 172 BPM and 44100 Hz sample rate
		HOP_SIZE = int(FRAME_SIZE/2)	# About 0.5 beat interval at 172 BPM 
		
		spec = Spectrum(size = FRAME_SIZE)
		specPeaks = SpectralPeaks()
		hpcp = HPCP()
		key = Key(profileType='edma')
		w = Windowing(type = 'blackmanharris92')
		fft = np.fft.fft
		pool = Pool()
		
		for frame in FrameGenerator(audio, frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
			frame_spectrum = spec(w(frame))
			frequencies, magnitudes = specPeaks(frame_spectrum)
			hpcpValue = hpcp(frequencies, magnitudes)
			pool.add('hpcp', hpcpValue)
			
		hpcp_avg = np.average(pool['hpcp'], axis=0)
		key,scale,_,_ = key(hpcp_avg)
		return key, scale
