# Copyright 2017 Len Vande Veire, IDLab, Department of Electronics and Information Systems, Ghent University
# This file is part of the source code for the Auto-DJ research project, published in Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system.", 2018 (submitted)
# Released under AGPLv3 license.

import song
from util import *
import os

import logging
logger = logging.getLogger('colorlogger')

circle_of_fifths = {
 	'major' : ['C','G','D','A' ,'E' ,'B' ,'F#','C#','G#','D#','A#', 'F'],
 	'minor' : ['A','E','B','F#','C#','G#','D#','A#','F' ,'C' ,'G' ,'D']		
}
# circle_of_fifths = {
#     'major' : ['C','G','D','A','E','B','Gb','Db','Ab','Eb','Bb','F'],
#     'minor' : ['A','E','B','F#','C#','G#','Eb','Bb','F','C','G','D']
#     }
notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def get_key(key, scale, offset, switchMajorMinor = False):
	# Get the key at key + offset around circle of fifths
	idx = (circle_of_fifths[scale].index(key) + offset ) % 12
	scale2 = scale if not switchMajorMinor else ('major' if scale == 'minor' else 'minor')
	return circle_of_fifths[scale2][idx], scale2
	
def get_key_transposed(key, scale, semitones):
	idx = notes.index(key)
	return notes[(idx + semitones)%12], scale
	
def get_relative_key(key, scale):
	if scale == 'major':
		new_key, _ = get_key_transposed(key, scale, -3)
		return new_key , 'minor'
	else:
		new_key, _ = get_key_transposed(key, scale, 3)
		return new_key, 'major'
	
def get_closely_related_keys(key, scale):
	result = []
	# Same key
	result.append((key, scale))
	# Relative key
	result.append(get_relative_key(key,scale))
	# Perfect fifth up
	result.append(get_key_transposed(key,scale,7))
	# Perfect fifth down
	result.append(get_key_transposed(key,scale,-7))
	return result
	
def distance_keys_semitones(key1,key2):
	idx1 = notes.index(key1)
	idx2 = notes.index(key2)
	return (idx2-idx1)%12
	
def distance_keys_circle_of_fifths(key1, scale1, key2, scale2):
	# Positive distance: clockwise
	# Negative distance: counterclockwise
	idx1 = circle_of_fifths[scale1].index(key1)
	idx2 = circle_of_fifths[scale2].index(key2)
	return ((6+((idx2-idx1)%12))%12)-6
	
class SongCollection:
	
	def __init__(self):
		self.songs = []			# A list holding all (annotated) songs
		self.directories = []	# A list containing all loaded directories
		self.key_title = {}
		
		self.vocal_songs = None
		
	def init_key_title_map(self):
		self.key_title = {}
		annotated_titles = [s.title for s in self.get_annotated()]
		for dir_ in self.directories:
			# Keep a list of the annotated songs and their keys
			songs_key_list = loadCsvAnnotationFile(dir_, ANNOT_KEY_PREFIX)
			for title, key in songs_key_list.items():
				if not title in annotated_titles:
					continue
				if not key in self.key_title:
					self.key_title[key] = [title]
				else:
					self.key_title[key].append(title)
		for key, songs in iter(sorted(self.key_title.items())):
			logger.info('Key {} :\t{} songs'.format(key, len(songs)))
	
	def load_directory(self, directory):
		directory_ = os.path.abspath(directory)
		if directory_ in self.directories:
			return				# Don't add the same directory twice
		logger.info('Loading directory ' + directory + '...')
		self.directories.append(directory_)
		self.songs.extend([song.Song(os.path.join(directory_, f)) for f in os.listdir(directory_) if os.path.isfile(os.path.join(directory_, f)) and (f.endswith('.wav') or f.endswith('.mp3'))])
		self.init_key_title_map()
		
	def annotate(self):
		for s in self.get_unannotated():
			s.annotate()
		self.init_key_title_map()
	
	def get_unannotated(self):
		return [s for s in self.songs if not s.hasAllAnnot()]
		
	def get_annotated(self):
		return [s for s in self.songs if s.hasAllAnnot()]
		
	def get_marked(self):
		markedTitles = []
		with open('markfile.csv') as csvfile:
			reader = csv.reader(csvfile)
			for line in reader:
				print(line)
				markedTitles.extend(line)
		return [s for s in self.songs if s.title in markedTitles]
		
	def get_titles_in_key(self, key, scale, offset=0, switchMajorMinor=False):
				
		# Return the songs that are in key with this song, one perfect fifth higher or one perfect fifth lower
		result = []
		key, scale = get_key(key, scale, offset, switchMajorMinor)
		try:
			titles_to_add = self.key_title[key+':'+scale]
			result += titles_to_add
		except KeyError:
			pass # No songs in this key
		return result
		
if __name__ == '__main__':
	
	from djcontroller import DjController
	from tracklister import TrackLister
	
	# Open the long library
	sc = SongCollection()
	sc.load_directory('../music/')
	sc.songs[0].open()
	logger.debug(sc.songs[0].tempo)
	# Generate a tracklist
	tl = TrackLister(sc)
	tl.generate(10)
	# Play!
	sm = DjController(tl)
	sm.play()
		
