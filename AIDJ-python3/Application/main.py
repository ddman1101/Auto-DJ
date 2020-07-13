# Copyright 2017 Len Vande Veire, IDLab, Department of Electronics and Information Systems, Ghent University
# This file is part of the source code for the Auto-DJ research project, published in Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system.", 2018 (submitted)
# Released under AGPLv3 license.

from songcollection import SongCollection
from tracklister import TrackLister
from djcontroller import DjController
import essentia
import os

import logging
#~ LOG_LEVEL = logging.INFO
LOG_LEVEL = logging.DEBUG
LOGFORMAT = "%(log_color)s%(message)s%(reset)s"
from colorlog import ColoredFormatter
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
logger = logging.getLogger('colorlogger')
logger.setLevel(LOG_LEVEL)
logger.addHandler(stream)

if __name__ == '__main__':
	
	sc = SongCollection()
	tl = TrackLister(sc)
	dj = DjController(tl)
	
	essentia.log.infoActive = False
	essentia.log.warningActive = False
	
	while(True):
		try:
			cmd_split = str.split(input('> : '), ' ')
		except KeyboardInterrupt:
			logger.info('Goodbye!')
			break
		cmd = cmd_split[0]
		if cmd == 'loaddir':
			if len(cmd_split) == 1:
				logger.warning('Please provide a directory name to load!')
				continue
			elif not os.path.isdir(cmd_split[1]):
				logger.warning(cmd_split[1] + ' is not a valid directory!')
				continue
			sc.load_directory(cmd_split[1])
			logger.info(str(len(sc.songs)) + ' songs loaded [annotated: ' + str(len(sc.get_annotated())) + ']')
		elif cmd == 'play':
			if len(sc.get_annotated()) == 0:
				logger.warning('Use the loaddir command to load some songs before playing!')
				continue
			
			if len(cmd_split) > 1 and cmd_split[1] == 'save':
				logger.info('Saving this new mix to disk!')
				save_mix = True
			else:
				save_mix = False
				
			logger.info('Starting playback!')
			try:
				dj.play(save_mix=save_mix)
			except Exception as e:
				logger.error(e)
		elif cmd == 'pause':
			logger.info('Pausing playback!')
			try:
				dj.pause()
			except Exception as e:
				logger.error(e)
		elif cmd == 'skip' or cmd == 's':
			logger.info('Skipping to next segment...')
			try:
				dj.skipToNextSegment()
			except Exception as e:
				logger.error(e)
		elif cmd == 'stop':
			logger.info('Stopping playback!')
			dj.stop()
		elif cmd == 'save':
			logger.info('Saving the next new mix!')
		elif cmd == 'showannotated':
			logger.info('Number of annotated songs ' + str(len(sc.get_annotated())))
			logger.info('Number of unannotated songs ' + str(len(sc.get_unannotated())))
		elif cmd == 'annotate':
			logger.info('Started annotating!')
			sc.annotate()
			logger.info('Done annotating!')
		elif cmd == 'debug':
			LOG_LEVEL = logging.DEBUG
			logging.root.setLevel(LOG_LEVEL)
			stream.setLevel(LOG_LEVEL)
			logger.setLevel(LOG_LEVEL)
			logger.debug('Enabled debug info. Use this command before playing, or it will have no effect.')
		elif cmd == 'mark':
			dj.markCurrentMaster()		
		else:
			logger.info('The command ' + cmd + ' does not exist!')
		
		
	
