# Copyright 2017 Len Vande Veire, IDLab, Department of Electronics and Information Systems, Ghent University
# This file is part of the source code for the Auto-DJ research project, published in Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system.", 2018 (submitted)
# Released under AGPLv3 license.

from essentia import *
import songcollection
from timestretching import time_stretch_sola, time_stretch_and_pitch_shift
import songtransitions 
from song import *
from util import *

import numpy as np
from scipy.spatial.distance import euclidean as euclidean_distance
import random

import logging
logger = logging.getLogger('colorlogger')

# ------------------- CONSTANTS AND PARAMETERS ------------------------

TYPE_DOUBLE_DROP = 'double drop'
TYPE_ROLLING = 'rolling'
TYPE_CHILL = 'relaxed'

TRANSITION_PROBAS = {
	TYPE_CHILL : [0.0, 0.7, 0.3],	# chill -> chill, rolling, ddrop
	TYPE_ROLLING : [0.2, 0.8, 0.0], # rolling> chill, rolling, ddrop
	TYPE_DOUBLE_DROP : [0.2, 0.8, 0.0] # ddrop -> chill, rolling, ddrop
	}

LENGTH_ROLLING_IN = 16
LENGTH_ROLLING_OUT = 16
LENGTH_DOUBLE_DROP_IN = LENGTH_ROLLING_IN
LENGTH_DOUBLE_DROP_OUT = 32
LENGTH_CHILL_IN = 16
LENGTH_CHILL_OUT = 16
	
THEME_WEIGHT = 0.4
PREV_SONG_WEIGHT = - 0.1 * (1-THEME_WEIGHT)
CURRENT_SONG_WEIGHT = 1 - (THEME_WEIGHT + PREV_SONG_WEIGHT)

NUM_SONGS_IN_KEY_MINIMUM = 15	# Used for song selection: the minimum number of songs that the tracklister attempts to add to the next song candidate pool
NUM_SONGS_ONSETS = 6			# Used for song selection: the number of songs of which the ODFs have to be matched with the master
MAX_SONGS_IN_SAME_KEY = 6		# Used for song selection: how many times songs in the same key can be played after each other

ROLLING_START_OFFSET = LENGTH_ROLLING_IN + LENGTH_ROLLING_OUT 	# Start N downbeats after previous queue point.

def is_vocal_clash_pred(master,slave):
	master = 2*master[1:-1] + master[:-2] + master[2:] >= 2
	slave = 2*slave[1:-1] + slave[:-2] + slave[2:] >= 2
	return sum(np.logical_and(master,slave)) >= 2
	
# Helper functions for getting the right downbeat positions		
def getDbeatAfter(song, dbeat, options, n=1):
	''' Get the nth segment downbeat after dbeat and after start_dbeat that is in options (dbeat not included)'''
	if dbeat is None:
		return None
	candidates = [b for b in options if b > dbeat]
	if len(candidates) < n:
		return None
	else:
		return candidates[n-1]
		
def getHAfter(song, dbeat, n=1):
	''' Get the first segment downbeat before dbeat and after start_dbeat that is H (dbeat not included) '''
	options = [song.segment_indices[i] for i in range(len(song.segment_indices)) if song.segment_types[i] == 'H']
	return getDbeatAfter(song,dbeat, options, n=n)
	
def getLAfter(song, dbeat, n=1):
	''' Get the first segment downbeat before dbeat and after start_dbeat that is L (dbeat not included) '''
	options = [song.segment_indices[i] for i in range(len(song.segment_indices)) if song.segment_types[i] == 'L']
	return getDbeatAfter(song, dbeat, options, n=n)

def getAllMasterSwitchPoints(song, fade_type):
	'''
		Returns tuple (switch_points, fade_in_lengths, fade_out_lengths)
		- switch_points : array of switch points in downbeats
		- fade_in_lengths : maximum allowed fade in length for this switch point for the master song
		- fade_out_lengths : maximum allowed fade out length for this switch point for the master song
	'''
	
	types, indices = song.segment_types, song.segment_indices
	LH = [indices[i] for i in range(1,len(indices)) if types[i-1] == 'L' and types[i] == 'H']
	HL = [indices[i] for i in range(1,len(indices)) if types[i-1] == 'H' and types[i] == 'L']

	if fade_type == TYPE_DOUBLE_DROP:
		# Return all L->H transitions
		cues = [i-1 for i in LH if i <= indices[-1] - LENGTH_DOUBLE_DROP_OUT]
		L_fade_in = [min(LENGTH_DOUBLE_DROP_IN, i - indices[0]) - 1 for i in cues]
		L_fade_out = [LENGTH_DOUBLE_DROP_OUT + 1 for i in cues]
		
	elif fade_type == TYPE_ROLLING:
		# Return all H->H->L transitions, i.e. a H->H transition, where the second H segment transitions to L after 16 measures
		cues = [i - LENGTH_ROLLING_OUT - LENGTH_ROLLING_IN - 1 for i in HL if i - LENGTH_ROLLING_OUT > 0 and i <= indices[-1] - LENGTH_ROLLING_OUT]
		L_fade_in = [min(LENGTH_ROLLING_IN, i - indices[0]) - 1 for i in cues]
		L_fade_out = [min(LENGTH_ROLLING_OUT, getLAfter(song, i) - i) + 1 for i in cues]
		
	elif fade_type == TYPE_CHILL:
		# Return all H->L transitions
		cues = HL
		L_fade_in = [min(LENGTH_CHILL_IN, i - indices[0]) for i in cues]
		L_fade_out = [min(LENGTH_CHILL_OUT, (getHAfter(song,i) if not (getHAfter(song,i) is None) else indices[-1]) - i) for i in cues]
	
	else:
		raise Exception('Unknown fade type {}'.format(fade_type))
	
	return list(zip(cues,L_fade_in, L_fade_out))
		
def getAllSlaveCues(song, fade_type, min_playable_length = 32):
	'''
		Returns tuple (switch_points, fade_in_lengths)
		- switch_points: the switch point in the slave
		- fade_in_lengths
		
		Fade out lengths are guaranteed by the min_playable_length argument
	'''
	
	types, indices = song.segment_types, song.segment_indices
	
	LH = [indices[i] for i in range(1,len(indices)) if types[i-1] == 'L' and types[i] == 'H' and indices[-1] - indices[i] > min_playable_length]
	HL = [indices[i] for i in range(1,len(indices)) if types[i-1] == 'H' and types[i] == 'L' and indices[-1] - indices[i] > min_playable_length]

	if fade_type == TYPE_DOUBLE_DROP:
		# Return all L->H transitions
		cues = [i-1 for i in LH]
		fade_in_lengths = [min(i, LENGTH_DOUBLE_DROP_IN-1) for i in cues]
	elif fade_type == TYPE_ROLLING:
		# Return all L->H transitions
		cues = [i-1 for i in LH]
		fade_in_lengths = [min(i, LENGTH_ROLLING_IN-1) for i in cues]
	elif fade_type == TYPE_CHILL:
		# Return beginning of the song
		cues = [indices[0] + LENGTH_CHILL_IN]
		fade_in_lengths = [LENGTH_CHILL_IN]
	else:
		raise Exception('Unknown fade type {}'.format(fade_type))
		
	return list(zip(cues, fade_in_lengths))
	

def getMasterQueue(song, start_dbeat, cur_fade_type):
	
	'''
		Get the (potential) next queue point
		Returns the queue point, fade type, maximum fade in length and fade out length
		
		start_dbeat: first downbeat from where fade can start (causality constraint)
		
		returns:
		- queue: the point of SWITCHING (because fade in length not known yet)
		- fade_type: double drop, rolling or chill?
	'''
	# Align start_dbeat with a phrase boundary before proceeding
	start_dbeat = start_dbeat + (8-(start_dbeat - song.segment_indices[0])%8)%8
	
	P_chill, P_roll, P_ddrop = TRANSITION_PROBAS[cur_fade_type]
	
	# Determine if this should (and can) be a double drop
	# If there are no 'H' segments anymore, then double drop is impossible
	# If the next 'H' segment is too late in the song (less than 32 downbeats before the last L segment), then this doesn't make sense either
	if P_ddrop > 0:
		
		isDoubleDrop = (random.random() <= P_ddrop)
		cues = getAllMasterSwitchPoints(song, TYPE_DOUBLE_DROP)
		cues = [c for c in cues if c[0] >= start_dbeat]
		
		if isDoubleDrop and len(cues) != 0:
			doubleDropDbeat, max_fade_in_len, fade_out_len = cues[0]
			max_fade_in_len = min(max_fade_in_len, doubleDropDbeat - start_dbeat - 1)
			return doubleDropDbeat - max_fade_in_len, TYPE_DOUBLE_DROP, max_fade_in_len, fade_out_len
	
	P_roll = P_roll / (P_roll + P_chill)
	P_chill = P_chill / (P_roll + P_chill)
	
	if P_roll > 0:	
		
		isRolling = (random.random() <= P_roll)
		cues = getAllMasterSwitchPoints(song, TYPE_ROLLING)
		cues = [c for c in cues if c[0] >= start_dbeat + ROLLING_START_OFFSET - 1]
		
		if isRolling and len(cues) != 0:
			rollingDbeat, max_fade_in_len, fade_out_len = cues[0]
			max_fade_in_len = min(max_fade_in_len, rollingDbeat - start_dbeat - 1)
			return rollingDbeat - max_fade_in_len, TYPE_ROLLING, max_fade_in_len, fade_out_len
			
	# No rolling transition or double drop: must be a chill transition
	if True:		# Only reason this is here is to have the same indentation as above :)
		if True:	
			# Transition point: first low segment after the first high segment (or this dbeat if it is H)
			# The song must play a bit before doing a chill transition!
			cues = getAllMasterSwitchPoints(song, TYPE_CHILL)
			cues = [c for c in cues if c[0] >= start_dbeat]
			if len(cues) == 0:
				cue, max_fade_in_len, fade_out_len = start_dbeat, 0, min(LENGTH_CHILL_OUT, song.segment_indices[-1] - start_dbeat)
			else:
				cue, max_fade_in_len, fade_out_len = cues[0]
			max_fade_in_len = min(max_fade_in_len, cue - start_dbeat)
			return cue - max_fade_in_len, TYPE_CHILL, max_fade_in_len, fade_out_len

def getSlaveQueue(song, fade_type, min_playable_length = 32):
	''' Search the slave song for a good transition point with type fade_type (chill, rolling, double drop) '''

	cues = getAllSlaveCues(song, fade_type, min_playable_length)
	
	if fade_type == TYPE_DOUBLE_DROP or fade_type == TYPE_ROLLING:
		if len(cues) > 0:
			cue, fade_in_len = cues[np.random.randint(len(cues))]	# -1 to switch just before drop
			return cue, fade_in_len
		else:
			cues = getAllSlaveCues(song, TYPE_CHILL, min_playable_length)
			logger.debug('Warning: no H dbeats!')
	
	# Choose the first L segment (beginning of song)
	cue, fade_in_len = cues[0]	
	return cue, fade_in_len

def calculateOnsetSimilarity(odf1, odf2):
	'''
		Compare two onset detection functions by means of the DTW algorithm.
		Beam search is used such that the maximum shift of the ODFs (in the DTW algorithm) is limited
	'''
	# ODF1 and ODF2 can differ in tempo and thus their ODF extracts can differ in length
	# This algorithm calculates the distance between the two ODFs by means of an adapted version of the DTW algorithm
	# It uses beam search, and offsets of +-N samples from the diagonal of the (rectangular) similarity matrix are allowed
	# We are also only interested in the final similarity score, not in the path
	
	# Make sure ODF1 is longer than ODF2
	if len(odf1) < len(odf2):
		temp = odf1
		odf1 = odf2
		odf2 = temp
	avg1 = np.average(odf1)
	# Scale ODFs so that they are comparable
	if avg1 != 0:
		odf1 /= avg1
	avg2 = np.average(odf2)
	if avg2 != 0:
		odf2 /= avg2
		
	N = 2
	scores = [0] * (2*N+1)
	prev_scores = scores
	prev_i2_center = 0
	slope = float(len(odf2))/len(odf1)
	
	for i1 in range(len(odf1)):
		i2_center = int(i1*slope+0.5) # Center of the [-N,N] beam around the diagonal
		for i in range(0,2*N+1):
			i2 = i2_center - N + i
			if i2 >= len(odf2) or i2 < 0:
				break
			score_increment = abs(odf1[i1] - odf2[i2])
			
			if prev_i2_center == i2_center:
				# Try to go right from previous (always possible, because centers are equal)
				score_new = prev_scores[i]
				if i > 0:
					# Try to go up from below
					score_new = min(score_new, scores[i-1])
					# Try to go diagonal
					score_new = min(score_new, prev_scores[i-1])
			else:
				# Try to go diagonal (always possible, because center has been shifted diagonally)
				score_new = prev_scores[i]
				# Try to go right
				if i < 2*N:
					score_new = min(score_new, prev_scores[i+1])
				# Try to go up from below
				if i > 0:
					score_new = min(score_new, scores[i-1])
			scores[i] = score_new + score_increment
		prev_i2_center = i2_center
		prev_scores = scores
	# Best score is in scores[N], i.e. on the diagonal
	return scores[N]
	
class TrackLister:
	
	def __init__(self, song_collection):
		self.songs = None 		# Ordered list of the songs in this tracklist
		self.crossfades = None	# Ordered list of crossfade objects: one less than number of songs
		self.song_collection = song_collection
		self.songsUnplayed = song_collection.get_annotated()	# Subset of song collection containing all unplayed songs
		self.songsPlayed = []									# List of songs already played
		self.song_file_idx = 0
		self.semitone_offset = 0
		
		self.theme_centroid = None
		self.prev_song_theme_descriptor = None
		
	def getFirstSong(self):
			
		# Do some initialization and return the first song to be played
		self.songsUnplayed = self.song_collection.get_annotated()	# Subset of song collection containing all unplayed songs
		firstSong = np.random.choice(self.songsUnplayed, size=1)[0]
		#~ firstSong = [s for s in self.songsUnplayed if 'Crude Tactics' in s.title][0]
		self.songsUnplayed.remove(firstSong)
		self.songsPlayed.append(firstSong)
		firstSong.open()
		
		# Choose a song to build up towards
		self.chooseNewTheme(firstSong)
		self.prev_song_theme_descriptor = firstSong.song_theme_descriptor
		
		return firstSong
		
	def chooseNewTheme(self, firstSong):
		# Initialize the theme centroid
		# 1. Measure the distance of each song to the first song
		songs_distance_to_first_song = []
		songs_themes = []
		for song in self.songsUnplayed:
			song.open()
			theme = song.song_theme_descriptor
			songs_themes.append(firstSong.song_theme_descriptor)
			songs_distance_to_first_song.append(euclidean_distance(theme, firstSong.song_theme_descriptor))
			song.close()
		songs_sorted = np.argsort(songs_distance_to_first_song)
		# 2. Select songs that are close to this first song
		indices_sorted = songs_sorted[:int(len(songs_sorted)/4)]
		# 3. Calculate the centroid of these songs
		self.theme_centroid = np.average(np.array(songs_themes)[indices_sorted],axis=0)
		
	def getSongOptionsInKey(self, key, scale):
		
		'''
			Returns a subset of the unplayed songs that are as much in key as possible with the given key and scale,
			and that attempt to build up towards the goal song its key and scale.
			The number of songs in the subset is at least NUM_SONGS_IN_KEY_MINIMUM
		'''
		songs_in_key = []
		songs_unplayed = self.songsUnplayed
		keys_added = []
		
		def addSongsInKey(key, scale):
			if (key,scale) not in keys_added:
				titles = self.song_collection.get_titles_in_key(key,scale)
				songs_to_add = [s for s in songs_unplayed if s.title in titles]
				songs_in_key.extend(songs_to_add)
				#~ logger.debug('{} songs in {} {} added'.format(len(songs_to_add), key, scale))
				keys_added.append((key,scale))
		
		closely_related_keys = songcollection.get_closely_related_keys(key, scale)
		for key_, scale_ in closely_related_keys:
			addSongsInKey(key_, scale_)
			# Also add keys one semitone higher or lower: these can be pitch shifted
			#~ key_to_add, scale_to_add = songcollection.get_key_transposed(key_, scale_, 1)
			#~ addSongsInKey(key_to_add, scale_to_add)
			key_to_add, scale_to_add = songcollection.get_key_transposed(key_, scale_, -1)
			addSongsInKey(key_to_add, scale_to_add)
			
		# 4: if still not enough songs, add random songs
		if len(songs_in_key) == 0:
			logger.debug('Not enough songs in pool, adding all songs!')
			songs_in_key = self.songsUnplayed			
		return np.array(songs_in_key) 
		
	def filterSongOptionsByThemeDistance(self, song_options, master_song):
		
		# First select NUM_SONGS_THEME songs that have a similar theme profile.
		# From this set of songs, the NUM_SONG_ONSETS most similar ones are selected and evaluated based on their onset similarity
		song_options_distance_to_centroid = []
		
		# Use this weighted average of the master song and goal song to select the next song
		cur_theme_centroid = (THEME_WEIGHT * self.theme_centroid 
			+ CURRENT_SONG_WEIGHT * master_song.song_theme_descriptor
			+ PREV_SONG_WEIGHT * self.prev_song_theme_descriptor)
		for song in song_options:
			song.open()
			dist_to_centroid = euclidean_distance(cur_theme_centroid, song.song_theme_descriptor)
			song_options_distance_to_centroid.append(dist_to_centroid)
			song.close()
		song_options_closest_to_centroid = np.argsort(song_options_distance_to_centroid)
		
		# Log messages
		logger.debug('Selected songs, ordered by theme similarity:')
		for i in song_options_closest_to_centroid[:NUM_SONGS_ONSETS]:
			song_options[i].open()
			title = song_options[i].title
			dist_to_centroid = song_options_distance_to_centroid[i]
			key = song_options[i].key
			scale = song_options[i].scale
			logger.debug('>> Theme difference {:20s} : {:.2f} ({} {})'.format(title[:20], dist_to_centroid, key, scale))
			song_options[i].close()
			
		return song_options[song_options_closest_to_centroid[:NUM_SONGS_ONSETS]]
		
	def getBestNextSongAndCrossfade(self, master_song, master_cue, master_fade_in_len, fade_out_len, fade_type):
		'''
			Choose a song that overlaps best with the given song
			The type of transition is also given (rolling, double drop, chill).
		'''
		transition_length = master_fade_in_len + fade_out_len
		
		# 1. Select songs that are similar in key and that build up towards the goal song
		key, scale = songcollection.get_key_transposed(master_song.key, master_song.scale, self.semitone_offset)
		song_options = self.getSongOptionsInKey(key, scale)
		closely_related_keys = songcollection.get_closely_related_keys(key, scale)
		
		# 2. Filter the songs in key based on their distance to the centroid
		song_options = self.filterSongOptionsByThemeDistance(song_options, master_song)
		#~ song_options = np.random.choice(song_options, size=NUM_SONGS_ONSETS)
			
		# 3. Filter based on vocal activity and ODF overlap
		master_song.open()
		best_score = np.inf
		best_score_clash = np.inf
		best_song = None
		logger.debug('Selected songs, evaluated by ODF similarity: ')
		for s in song_options:
			# Open the song
			next_song = s
			next_song.open()
			
			# Determine the queue points for the current song
			queue_slave, fade_in_len = getSlaveQueue(next_song, fade_type, min_playable_length = transition_length + 16)
			fade_in_len = min(fade_in_len, master_fade_in_len)
			fade_in_len_correction = master_fade_in_len - fade_in_len
			master_cue_corr = master_cue + fade_in_len_correction
			transition_len_corr = transition_length - fade_in_len_correction
			queue_slave = queue_slave - fade_in_len
			
			# Construct the cross-fade for this transition
			if queue_slave >= 16:
				cf = songtransitions.CrossFade(0, [queue_slave], transition_len_corr, fade_in_len, fade_type)
			else:
				cf = songtransitions.CrossFade(0, [queue_slave], transition_len_corr, fade_in_len, fade_type)
				
			# Iterate over the different options for queue_slave
			for queue_slave_cur in cf.queue_2_options:
				
				# Split the overlapping portions of the onset curves in segments of 4 downbeats
				# and calculate the similarities. The most dissimilar segment indicates the overall quality of the crossfade
				
				odf_segment_len = 4 # dbeats
				odf_scores = []
				for odf_start_dbeat in range(0, transition_len_corr, odf_segment_len):
					odf_master = master_song.getOnsetCurveFragment(master_cue_corr + odf_start_dbeat, min(master_cue_corr + odf_start_dbeat+odf_segment_len, master_cue_corr + transition_len_corr))
					odf_slave = s.getOnsetCurveFragment(queue_slave_cur + odf_start_dbeat, min(queue_slave_cur+odf_start_dbeat+odf_segment_len, queue_slave_cur+transition_len_corr))
					onset_similarity = calculateOnsetSimilarity(odf_master,odf_slave) / odf_segment_len
					odf_scores.append(onset_similarity)
				
				singing_scores = []
				singing_master = np.array(master_song.singing_voice[master_cue_corr : master_cue_corr + transition_len_corr] > 0)
				singing_slave = np.array(s.singing_voice[queue_slave : queue_slave + transition_len_corr] > 0)
				singing_clash = is_vocal_clash_pred(singing_master, singing_slave)
				
				onset_similarity = np.average(odf_scores)
				score = onset_similarity
				
				if score < best_score and not singing_clash:
					best_song = next_song
					best_score = score
					best_fade_in_len = fade_in_len
					best_slave_cue = queue_slave_cur
					best_master_cue = master_cue_corr
				elif best_score == np.inf and score < best_score_clash and singing_clash:
					best_song_clash = next_song
					best_score_clash = score
					best_fade_in_len_clash = fade_in_len
					best_slave_cue_clash = queue_slave_cur
					best_master_cue_clash = master_cue_corr
				
				type_fade_dbg_str = '>> {:20s} [{}:{:3d}]: ODF {:.2f} {}'.format(
					next_song.title[:20], 
					fade_type, 
					queue_slave_cur, 
					score, 
					'' if not singing_clash else '>>CLASH<<'
					)
					
				# Logging
				logger.debug(type_fade_dbg_str)
		
		if best_song is None:
			# No best song without vocal clash was found: use the clashing version instead as a last resort
			best_song = best_song_clash
			best_score = best_score_clash
			best_fade_in_len = best_fade_in_len_clash
			best_slave_cue = best_slave_cue_clash
			best_master_cue = best_master_cue_clash
		
		# Determine the pitch shifting factor for the next song
		key_distance = abs(songcollection.distance_keys_semitones(key, best_song.key))
		if (best_song.key, best_song.scale) not in closely_related_keys:
			# This song has been shifted one semitone up or down: this has to be compensated by means of pitch shifting
			shifted_key_up, shifted_scale_up = songcollection.get_key_transposed(best_song.key, best_song.scale, 1)
			if (shifted_key_up, shifted_scale_up) in closely_related_keys:
				self.semitone_offset = 1
			else:
				self.semitone_offset = -1
			logger.debug('Pitch shifting! {} {} by {} semitones'.format(best_song.key, best_song.scale, self.semitone_offset))
		else:
			self.semitone_offset = 0
			
		self.prev_song_theme_descriptor = master_song.song_theme_descriptor
		self.songsPlayed.append(best_song)
		self.songsUnplayed.remove(best_song)
		if len(self.songsUnplayed) <= NUM_SONGS_IN_KEY_MINIMUM: # If there are too few songs remaining, then restart
			logger.debug('Replenishing song pool')
			self.songsUnplayed.extend(self.songsPlayed)
			self.songsPlayed = []	
			
		return best_song, best_slave_cue, best_master_cue, best_fade_in_len, self.semitone_offset
