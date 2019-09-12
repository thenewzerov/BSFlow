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
Yell at me if I haven't added the usage part here yet
```

##### Using the pre-trained models:
Create a folder with your .ogg or .egg file. This will be your target directory.

The final output file will be written in this directory.

The longest part is generally reading in and normalizing the audio.

##### Training your own models:
Take the beat saber songs you want to use to create input data from and copy the folders to another directory.
**DO NOT** use the data directory from the Beat Saber folder.  
I will **NOT** be responsible if you have to re-download all your songs.

Rename the files in the `models` folder (or delete, but if it doesn't work you'll have to checkout again).

# Development

**Note for other developers:**

Sorry for my Python!  Not my first language, and you can probably tell I'm coming from Java and C#

Logically, BSFlow breaks the actual song generation up into several different neural networks.  
Each of these is responsible for a different part of the generation (Y/N, Red-Blue-Both, Cut Direction, etc...)

When running, you may notice that with the default settings, the TensorFlow accuracy never gets that high.
This is on purpose.  Otherwise, we would always generate the same songs.

This can lead to problems however.  If your training doesn't produce decent outputs, 
try to adjust the values related to the specific network you want to tweak, and run it again.

For example, if you're not generating enough notes, 
try increasing the `zero_value_dropout`, deleting that model and re-creating the yn network.


