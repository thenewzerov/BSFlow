# BSFlow Song Generator
A Beat Saber song generator using Tensorflow

# Setup
This was developed using Python 3 and Tensorflow 2.0-rc. Make sure the following are installed:

1) Python 3
2) pip3

```bash
# clone the repo
$ git clone https://github.com/thenewzerov/bsflow.git

# change the working directory to BSFlow
$ cd bsflow

# install the requirements
$ python3 -m pip install -r requirements.txt
```

# Usage

```bash
usage: bsflow.py [-h] [--force-regen]
                 input directory bpm notes_per_beat difficulty iterations

Generate a Beat Saber Song from a .ogg/.egg file.

positional arguments:
  input           Audio input file (.ogg or .egg)
  directory       Directory containing directories of Beat Saber songs to
                  train the networks with
  bpm             Beats Per Minute for the final song and the training
  notes_per_beat  Maximum numbner of notes per beat
  difficulty      Target Difficulty for the final song
  iterations      Training Iterations for the networks

optional arguments:
  -h, --help      show this help message and exit
  --force-regen   Force the models and song data to be recreated.
```

A typical usage might be something like this:

```bash
python bsflow.py /path/to/my-song/music-file.ogg /path/to/my/training/ 200 4 Expert 500
```

The iterations value doesn't really matter if you're not training new data.

##### Using the pre-trained models:
Create a folder with your .ogg or .egg file. This will be your target directory.

The final output file will be written in this directory.

The longest part is generally reading in and normalizing the audio.

##### Training your own models:

**PLEASE READ THIS WHOLE SECTION BEFORE TRAINING**

Take the beat saber songs you want to use as training examples and copy the folders to another directory.
**DO NOT** use the data directory from the Beat Saber folder.  
I will **NOT** be responsible if you have to re-download all your songs.

Rename the files in the `models` folder (or delete, but if it doesn't work you'll have to checkout again).

Depending on how many songs you include in the folder, training can take a VERY long time, especially on the first run through.
As the training progresses, the details for each song are written to files so that the song data doesn't have to be completely
rebuilt every time.  These files are re-used several times if you train with the same settings again.

Training data is saved based on difficulty, beats per minute, and notes per beat. Changing any of these values will mean you need
to re-create all the data, so try with a small set of songs first to see if you're happy with the times and note spacing.

Loading and re-training data is a slow process.  Files are loaded and trained one at a time to reduce memory usage, so if you
REALLY want to train on a big data set, you'll probably want to leave it running overnight.

For reference, I've trained the included models on around 100 songs, and a few thousand training iterations.

# Development

**Note for other developers:**

Sorry for my Python!  Not my first language, and you can probably tell I'm coming from Java and C#

Logically, BSFlow breaks the actual song generation up into several different neural networks.  
Each of these is responsible for a different part of the generation (Y/N, Red-Blue-Both, Cut Direction, etc...)

When running, you may notice that with the default settings, the TensorFlow accuracy never gets that high.
This is sort of intentional.  It helps to generate more "unique" songs.

This can lead to problems however.  If your training doesn't produce decent outputs, 
try to adjust the values related to the specific network you want to tweak, and run it again.


