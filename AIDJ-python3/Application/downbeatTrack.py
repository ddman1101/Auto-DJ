'''
    File name: loopextractor.py
    Author: Jordan B. L. Smith
    Date created: 2 December 2019
    Date last modified: 18 December 2019
    License: GNU Lesser General Public License v3 (LGPLv3)
    Python Version: 3.7
'''

import librosa
import madmom
import numpy as np



def get_downbeats(signal):

    act = madmom.features.downbeats.RNNDownBeatProcessor()(signal)
    proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    processor_output = proc(act)
    downbeat_times = processor_output[processor_output[:,1]==1,0]
    return downbeat_times

if __name__ == "__main__":
    audio_path = './test.wav'
    get_downbeats(audio_path)
