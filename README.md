# Auto-DJ
This repository contains the source code of the automatic DJ system developed by Len Vande Veire, under the supervision of prof. Tijl De Bie. It had been designed for Drum and Bass music, a sub-genre of Electronic Dance Music. Then Wei-Han Hsu was changed the code from python2 to python3, and use madmom to change the part of downbeats. And he changed the genre from Drum and Bass to EDM music.

The system is described in more detail in the paper _Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system." 2018 (submitted)_.

## Problem

In the following some packages, it only works on the ubuntu system. So in this branch I will focus on fix these problems (on-going)

## Installing dependencies

The automatic DJ application has been developed for python 3.7.7 and tested on Ubuntu 18.04 LTS. It depends on the following python packages:

* colorlog (2.10.0)
* Essentia
* joblib (0.11)
* librosa (0.5.0)
* numpy (1.19.1)
* pyAudio (0.2.8)
* scikit-learn (0.18.1)
* scipy (0.19.0)
* yodel (0.3.0)
* madmom (0.17.dev0)

These packages can be installed using e.g. the `pip` package manager or using `apt-get` on Ubuntu. Installation instructions for the Essentia library can be found on [http://essentia.upf.edu/documentation/installing.html](http://essentia.upf.edu/documentation/installing.html).

You can also run 
`pip install -r requirements.txt`
to make the part of installing dependencies.

## Running the application

To run the application, run the `main.py` script in the `AIDJ-python3/Application` directory:

`python main.py`

The application is controlled using commands. The following commands are available:

* `loaddir <directory>` : Add the _.wav_ and _.mp3_ audio files in the specified directory to the pool of available songs.
* `annotate` : Annotate all the files in the pool of available songs that are not annotated yet. Note that this might take a while, and that in the current prototype this can only be interrupted by forcefully exiting the program (using the key combination `Ctrl+C`).
* `play` : Start a DJ mix. This command must be called after using the `loaddir` command on at least one directory with some annotated songs. Also used to continue playing after pausing. ( run : "play save" can save the set)
* `pause` : Pause the DJ mix.
* `stop` : Stop the DJ mix.
* `skip` : Skip to the next important boundary in the mix. This skips to either the beginning of the next crossfade, the switch point of the current crossfade or the end of the current crossfade, whichever comes first.
* `s` : Shorthand for the skip command
* `showannotated` : Shows how many of the loaded songs are annotated.
* `debug` : Toggle debug information output. This command must be used before starting playback, or it will have no effect.

To exit the application, use the `Ctrl+C` key combination.

## Some problem in the auto-dj
Sometimes the key of music may not in the list, then it couldn't play.

## Copyright information
Copyright 2017 Len Vande Veire, IDLab, Department of Electronics and Information Systems, Ghent University.

This file is part of the source code for the Auto-DJ research project, published in Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system." 2018 (submitted).

Released under AGPLv3 license.
