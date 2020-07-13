# Copyright 2017 Len Vande Veire, IDLab, Department of Electronics and Information Systems, Ghent University
# This file is part of the source code for the Auto-DJ research project, published in Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system.", 2018 (submitted)
# Released under AGPLv3 license.

'''	
	Standalone GUI tool to fix the annotations of songs.
	When the master song in the applications has a wrong annotation, use the 'mark' command to add it to the marked list.
	In this application, you can quickly edit those wrong annotations.
	Usage: python ToolFixAnnotations.py path_to_music_directory_1 path_to_music_directory_2 ...
'''


from song import Song
from songcollection import SongCollection
import sys
import numpy as np
import csv
from util import *

from essentia import *
from essentia.standard import *
import pyaudio

import tkinter as tk		   #This interface allow us to draw windows

def overlayAudio(audio, beats):
	onsetMarker = AudioOnsetsMarker(onsets = 1.0*beats)
	audioMarked = onsetMarker(audio)
	return audioMarked
	
class ToolFixAnnotationApp(tk.Tk):
	
	def __init__(self, directories, *args, **kwargs):
		print(args, kwargs)
		tk.Tk.__init__(self, *args, **kwargs)
		
		# Stream for audio playback
		self.pa = pyaudio.PyAudio()
		self.stream = self.pa.open(format = pyaudio.paFloat32,
							channels=1,
							rate=44100,
							output=True)
		self.protocol("WM_DELETE_WINDOW", self.close_window)
		
		sc = SongCollection()
		print(directories)
		for dir_ in directories:
			sc.load_directory(dir_)
		self.songs = sc.get_marked()
		self.song = None
		
		self.varSongSelector = tk.Variable(self)
		self.songSelector = tk.OptionMenu(self, self.varSongSelector, '', *tuple([s.title for s in self.songs]), command = self.select_song)
		self.songSelector.grid(row=0,columnspan=3)
		
		tk.Label(self, text="Beats: ").grid(row=1)
		self.b_beats_play = tk.Button(self, text="[ > ]", command=self.play_beats)
		self.b_beats_play.grid(row=1,column=1)
		self.b_beats_fix = tk.Button(self, text="[+1]", command=self.shift_beats)
		self.b_beats_fix.grid(row=1,column=2)
		
		tk.Label(self, text="Downbeats: ").grid(row=2)
		self.b_downbeats_play = tk.Button(self, text="[ > ]", command=self.play_downbeats)
		self.b_downbeats_play.grid(row=2,column=1)
		self.b_downbeats_fix = tk.Button(self, text="[+1]", command=self.shift_downbeats)
		self.b_downbeats_fix.grid(row=2,column=2)
		
		tk.Label(self, text="Segments: ").grid(row=3)
		self.b_segments_shift_min8 = tk.Button(self, text="[-8]", command=lambda : self.shift_segments(-8))
		self.b_segments_shift_min8.grid(row=3,column=1)
		self.b_segments_shift_min1 = tk.Button(self, text="[-1]", command=lambda : self.shift_segments(-1))
		self.b_segments_shift_min1.grid(row=3,column=2)
		self.b_segments_shift_plus1 = tk.Button(self, text="[+1]", command=lambda : self.shift_segments(1))
		self.b_segments_shift_plus1.grid(row=3,column=3)
		self.b_segments_shift_plus8 = tk.Button(self, text="[+8]", command=lambda : self.shift_segments(8))
		self.b_segments_shift_plus8.grid(row=3,column=4)
		
		self.segment_buttons = []
		
		# TODO How to insert segments?
		
		self.b_save = tk.Button(self, text="[ SAVE CHANGES ]", command=self.save)
		self.b_save.grid(row=0,column=4)
		
		#~ self.statusbar.pack(side="bottom", fill="x")
		#~ self.toolbar.pack(side="top", fill="x")
		#~ self.navbar.pack(side="left", fill="y")
		#~ self.main.pack(side="right", fill="both", expand=True)
		
	def select_song(self, title):
		if self.song != None:
			self.song.close()
		self.song_title = title
		self.song = [s for s in self.songs if s.title == title][0]
		self.song.open()
		self.song.openAudio()
		self.add_segment_buttons()
			
	def play_beats(self):
		print('Beats of ' + self.song.title)
		song = self.song
		high_segments = [i for i,j in zip(song.segment_indices, song.segment_types) if j == 'H']
		start_beat = song.downbeats[high_segments[0]]
		start_idx = int(song.downbeats[high_segments[0]-1]*44100)
		end_idx = int(song.downbeats[high_segments[0]+2]*44100)
		audio = song.audio[start_idx:end_idx]
		
		audioMarked = overlayAudio(audio, np.array([b-start_beat for b in song.beats if b >= start_beat],dtype='single'))
		self.stream.write(audioMarked, num_frames=len(audioMarked), exception_on_underflow=False)
		
	def shift_beats(self):
		song = self.song
		IBI = 60.0/song.tempo
		phase = song.phase
		song.phase = phase - IBI/2 if phase >= IBI/2 else phase + IBI/2
		song.beats = [b - phase + song.phase for b in song.beats]
		song.downbeats = [b - phase + song.phase for b in song.downbeats]
		
	def play_downbeats(self):
		print('Downbeats of ' + self.song.title)
		song = self.song
		high_segments = [i for i,j in zip(song.segment_indices, song.segment_types) if j == 'H']
		start_beat = song.downbeats[high_segments[0]]
		start_idx = int(song.downbeats[high_segments[0]-1]*44100)
		end_idx = int(song.downbeats[high_segments[0]+2]*44100)
		audio = song.audio[start_idx:end_idx]
		
		audioMarked = overlayAudio(audio, np.array([b-start_beat for b in song.downbeats if b >= start_beat],dtype='single'))
		self.stream.write(audioMarked, num_frames=len(audioMarked), exception_on_underflow=False)
		
	def shift_downbeats(self):
		song = self.song
		dbindex = song.beats.index(song.downbeats[0])%4
		song.downbeats = [song.beats[i] for i in range(len(song.beats)) if (i-(dbindex+1))%4 == 0]

	def play_segment(self,segidx):
		print('Playing segment ' + str(segidx))
		song = self.song
		
		start_idx = int(song.downbeats[segidx]*44100)
		end_idx = int(song.downbeats[segidx+4]*44100)
		audio = song.audio[start_idx:end_idx]
		
		self.stream.write(audio, num_frames=len(audio), exception_on_underflow=False)
		
	def shift_segments(self, shift):
		print('Shifting segments with factor: {}'.format(shift))
		print(self.song.segment_indices)
		segment_indices_new = [i+shift for i in self.song.segment_indices[:-1]]
		if shift < 0:
			segment_indices_new.append(self.song.segment_indices[-1]+shift)
		else:
			if self.song.segment_indices[-1]+shift < (self.song.downbeats[-1]):
				segment_indices_new.append(self.song.segment_indices[-1]+shift)
			else:
				new_ending = self.song.segment_indices[-1]+shift-8
				if new_ending not in segment_indices_new:
					segment_indices_new.append(new_ending)
		self.song.segment_indices = segment_indices_new
		self.song.loadAnnotSegments_fixNegativeStart()
		self.add_segment_buttons()
		print(self.song.segment_indices)
		
	def shift_segment(self, idx, shift):
		print(self.song.segment_indices)
		
		segment_index = self.song.segment_indices[idx] + shift
		
		if shift < 0:
			if idx == 0 or self.song.segment_indices[idx-1] < segment_index:
				self.song.segment_indices[idx] = segment_index
			else:
				print('Cannot shift segment beyond predecessor!')
		else:
			if idx < len(self.song.segment_indices)-1 and self.song.segment_indices[idx+1] > segment_index:
				self.song.segment_indices[idx] = segment_index
			elif idx == len(self.song.segment_indices)-1 and segment_index < len(self.song.downbeats):
				self.song.segment_indices[idx] = segment_index
			else:
				print('Cannot shift segment beyond successor or end of song!')
		
		segidx = self.song.segment_indices[idx]
		segtype = self.song.segment_types[idx]
		self.segment_buttons[4*idx]['text'] = '[{}:{}]'.format(segidx,segtype)
		self.segment_buttons[4*idx]['command'] = lambda i=segidx: self.play_segment(i)
		self.song.loadAnnotSegments_fixNegativeStart()
		
		print(self.song.segment_indices)
		
	def change_segment_type(self, idx):
		self.song.segment_types[idx] = 'H' if self.song.segment_types[idx] == 'L' else 'L'
		segidx = self.song.segment_indices[idx]
		segtype = self.song.segment_types[idx]
		self.segment_buttons[4*idx]['text'] = '[{}:{}]'.format(segidx,segtype)
		print(self.song.segment_indices)
		print(self.song.segment_types)
		
	def add_segment_buttons(self):
		for b in self.segment_buttons:
			b.grid_forget()
		self.segment_buttons = []
			
		for i, segidx, segtype in zip(list(range(len(self.song.segment_indices))), self.song.segment_indices, self.song.segment_types):
			# Add a label
			b = tk.Button(self, text="[{}:{}]".format(segidx,segtype), command=lambda i=segidx: self.play_segment(i))
			self.segment_buttons.append(b)
			b.grid(row=4+i, column=0)
			# Add some buttons to shift left and right
			b = tk.Button(self, text="[-1]", command=lambda i=i: self.shift_segment(i, -1))
			self.segment_buttons.append(b)
			b.grid(row=4+i, column=1)
			b = tk.Button(self, text="[+1]", command=lambda i=i: self.shift_segment(i, +1))
			self.segment_buttons.append(b)
			b.grid(row=4+i, column=2)
			# Add some buttons to change the label type
			b = tk.Button(self, text="[L/H]", command=lambda i=i: self.change_segment_type(i))
			self.segment_buttons.append(b)
			b.grid(row=4+i, column=3)
		
	def save(self):
		song = self.song
		print('Saving annotations...')
		writeAnnotFile(song.dir_, song.title, ANNOT_BEATS_PREFIX, song.beats, {'tempo' : song.tempo , 'phase' : song.phase})
		writeAnnotFile(song.dir_, song.title, ANNOT_DOWNB_PREFIX, song.downbeats)
		writeAnnotFile(song.dir_, song.title, ANNOT_SEGMENT_PREFIX, list(zip(song.segment_indices, song.segment_types)))	
		print('Annotations saved!')
		
	def close_window(self):
		self.stream.stop_stream()
		self.stream.close()
		self.pa.terminate()
		print('Closing window...')
		self.destroy()
	

if __name__ == '__main__':
	
	
	bla = ToolFixAnnotationApp(sys.argv[1:])
	bla.mainloop()
