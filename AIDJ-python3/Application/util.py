# Copyright 2017 Len Vande Veire, IDLab, Department of Electronics and Information Systems, Ghent University
# This file is part of the source code for the Auto-DJ research project, published in Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system.", 2018 (submitted)
# Released under AGPLv3 license.

import os, csv
import numpy as np
from essentia import *
from essentia.standard import *

ANNOT_SUBDIR = '_annot_auto/'		# Not _annot_... because we don't want to overwrite the existing (ground-truth) annotations!
ANNOT_DOWNB_PREFIX = 'downbeats_'
ANNOT_BEATS_PREFIX = 'beats_'
ANNOT_SEGMENT_PREFIX = 'segments_'
ANNOT_GAIN_PREFIX = 'gain_'
ANNOT_KEY_PREFIX = 'key_'
ANNOT_SPECTRALCONTRAST_PREFIX = 'specctrst_'
ANNOT_SINGINGVOICE_PREFIX = 'singing_'
ANNOT_ODF_HFC_PREFIX = 'odf_hfc_'

ANNOT_MARKED_PREFIX = '_fix_annotations_'

import logging
logger = logging.getLogger('colorlogger')

def pathAnnotationFile(directory, song_title, prefix):
	return os.path.join(directory, ANNOT_SUBDIR, prefix + song_title + '.txt')

def loadCsvAnnotationFile(directory, prefix):
	result = {}
	try:
		with open(os.path.join(directory, ANNOT_SUBDIR, prefix + '.csv'), 'r+') as csvfile:
			reader = csv.reader(csvfile, delimiter = ' ')
			for row in reader:
				key, value = row
				try:
					value = float(value)
				except ValueError:
					pass
				result[key] = value
	except IOError as e:
		logger.debug('Csv annotation file not found, silently ignoring exception ' + str(e))
	return result
	
def writeCsvAnnotation(directory, prefix, song_title, value):
	with open(os.path.join(directory, ANNOT_SUBDIR, prefix + '.csv'), 'a+') as csvfile:
		writer = csv.writer(csvfile, delimiter = ' ')
		if type(value) is float:
			writer.writerow([song_title, '{:.9f}'.format(value)])
		else:
			writer.writerow([song_title, '{:}'.format(value)])
			
def deleteCsvAnnotation(directory, prefix, song_title):
	# Delete one line in the CSV annotation file.
	# Warning: this will not work with very big files, because they won't fit in RAM
	titles = []
	with open(os.path.join(directory, ANNOT_SUBDIR, prefix + '.csv'), 'r+') as csvfile:
		reader = csv.reader(csvfile, delimiter = ' ')
		for line in reader:
			titles.append(line)
	with open(os.path.join(directory, ANNOT_SUBDIR, prefix + '.csv'), 'w+') as csvfile:
		writer = csv.writer(csvfile, delimiter = ' ')
		for line in titles:
			if line[0] != song_title:
				writer.writerow(line)

def loadAnnotationFile(directory, song_title, prefix):
	'''
	Loads an input file with annotated times in seconds.
	
	-Returns: A numpy array with the annotated times parsed from the file.
	'''
	input_file = pathAnnotationFile(directory, song_title, prefix)
	result = []
	result_dict = {}
	if os.path.exists(input_file):
		with open(input_file) as f:
			for line in f:
				if line[0] == '#':
					try:
						key, value = str.split(line[1:], ' ')
						result_dict[key] = float(value)
					except ValueError:
						# "too many values to unpack" because it's a normal comment line
						pass
				else:
					result.append(line)
	else:
		raise UnannotatedException('Attempting to load annotations of unannotated audio' + input_file + '!')
	return result, result_dict
	
def writeAnnotFile(directory, song_title, prefix, array, values_dict = {}):
	
	output_file = pathAnnotationFile(directory, song_title, prefix)	
	with open(output_file, 'w+') as f:
		# Write the dict
		for key, value in values_dict.items():
			f.write('#' + str(key) + ' ' + '{:.9f}'.format(value) + '\n')
		# Write the annotations
		for value in array:
			if type(value) is tuple:
				for v in value:
					if type(v) is float:
						f.write('{:.9f} '.format(v))
					else:
						f.write('{} '.format(v))
				f.write('\n')
			else:
				f.write("{:.9f}".format(value) + '\n')

def loadBinaryAnnotFile(directory, song_title, prefix):
	input_file = pathAnnotationFile(directory, song_title, prefix)
	result = []
	if os.path.exists(input_file):
		with open(input_file, mode='rb') as f:
			result = np.load(f, allow_pickle=True)
	else:
		raise UnannotatedException('Attempting to load annotations of unannotated audio' + input_file + '!')
	return result

def writeBinaryAnnotFile(directory, song_title, prefix, array):
	output_file = pathAnnotationFile(directory, song_title, prefix)	
	with open(output_file, 'wb+') as f:
		# Write the annotations
		np.save(f, array)
