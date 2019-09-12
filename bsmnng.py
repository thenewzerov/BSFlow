from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
from glob import glob
import soundfile as sf
import os
import json
import math
import numpy as np
import tensorflow as tf
import sys
from progress.bar import Bar
from os import path
import random

# Global Values
regen_nn_models = True
regen_files = True
normalization_accuracy = 2
bpm = 200
division_factor = 4
divisions_increment = .25
zero_beat_buffer = 15
slice_input = -1  # Set to -1 to not cut input size down.  Used for testing
difficulty = 'Expert'

# Y/N Model Values
yn_training_iterations = 5000
yn_input_size = 14
yn_output_size = 2
yn_zero_value_dropout = 70  # Dropout value for zero's so we don't over-fit no notes

# RBB Model Values
rbb_training_iterations = 5000
rbb_input_size = 16
rbb_output_size = 3
rbb_both_dropout = 57

# Cut Direction Model Values
cd_training_iterations = 100
cd_input_size = 16
cd_output_size = 4


# ----------------------------------------- General Methods ------------------------------------------------------#
def normalize_audio_file(file_path):
    """

    :param file_path:
    :return:
    """
    print("    Normalizing ", file_path)

    with sf.SoundFile(file_path, 'r') as f:
        data = f.read()

        beats_in_song = round(((len(f) / f.samplerate) / 60) * bpm)
        frames_per_beat = round(len(f) / (beats_in_song * division_factor))
        print('    Samples         = {}'.format(len(f)))
        print('    Sample Rate     = {}'.format(f.samplerate))
        print('    Seconds         = {}'.format(len(f) / f.samplerate))
        print('    Beats In  Song  = {} (at {} BPM)'.format(beats_in_song, bpm))
        print('    Frames Per Beat = {}'.format(frames_per_beat * division_factor))

        # Create an array to store the normalized data
        normalized =[]

        beat_sum = 0
        frame_count = 0
        min_frame = sys.maxsize
        max_frame = 0

        if path.exists(os.path.join(file_path + '.json')) and not regen_files:
            with open(os.path.join(file_path + '.json')) as json_file:
                normalized = json.load(json_file)
                json_file.close()
                if slice_input > -1:
                    del normalized[slice_input:]
                return np.array(normalized)

        # Frame Counter
        with Bar('Normalizing Song', max=len(data)) as bar:
            for x in range(len(data)):
                for c in range(f.channels):
                    if data[x][c] < min_frame:
                        min_frame = data[x][c]
                    if data[x][c] > max_frame:
                        max_frame = data[x][c]

                    beat_sum += data[x][c]

                frame_count += 1

                if frame_count == frames_per_beat:
                    average = (beat_sum / frames_per_beat)
                    normalized.append([min_frame, average, max_frame])
                    beat_sum = 0
                    frame_count = 0
                    min_frame = sys.maxsize
                    max_frame = 0
                bar.next()

        normalized = np.array(normalized)
        normalized = normalized - normalized.mean()
        normalized = normalized / np.abs(normalized).max()

        print('   Total Beats = {}'.format(len(normalized)))

    output_array = normalized.tolist()
    f = open(file_path + ".json", 'w')
    json.dump(output_array, f)
    f.close()

    print("    Normalizing complete")

    if slice_input > -1:
        del normalized[slice_input:]

    return normalized


# Build the song data for a normalized song and a beat saber .dat file.  Normalizes the note times
def get_notes_for_song(beat_saber_song_dat, original_bpm):
    print("Calculating Note Times")

    converted_notes = []

    if '_notes' in beat_saber_song_dat:

        notes = beat_saber_song_dat['_notes']
        notes = sorted(notes, key=lambda nt: nt['_time'])

        for note in notes:
            if note['_type'] == 0 or note['_type'] == 1:
                note['_time'] = x_round((note['_time'] / original_bpm) * bpm)
                converted_notes.append(note)

    print("Notes converted")

    return converted_notes


# Gets the metadata and the song data for a given directory
def get_data_for_directory(directory):
    normalized = None
    song_meta_data = None
    song_data = []

    print(" Building Training Data for ", directory)

    # Normalize the song data
    for file in os.listdir(directory):
        if file.endswith(".ogg") or file.endswith(".egg"):
            normalized = normalize_audio_file(os.path.join(directory, file))

    # Might as well do a sanity check to make sure we found a sound file
    if normalized is not None:

        # Load the original Song metadata so we can get the beats per minute
        for file in os.listdir(directory):
            if file.endswith(".dat") and file.startswith('info'):
                with open(os.path.join(directory, file)) as info_file:
                    song_meta_data = json.load(info_file)

        # Make sure we got metadata!
        if song_meta_data is not None:
            for file in os.listdir(directory):
                if file.endswith(difficulty + ".dat") and not file.startswith('info'):
                    with open(os.path.join(directory, file)) as json_file:
                        beat_saber_song_dat = json.load(json_file)
                        song_data.append(beat_saber_song_dat)

    print(' ', directory, " completed")
    return song_meta_data, normalized, song_data


# Build the step input data, last two values will not be filled out
def build_step_input(normalized_song_data, total_size):
    step_data = []

    print("    Creating Step Data for song")

    for step in range(len(normalized_song_data)):

        step_input = [0] * total_size

        # Set Previous Song levels
        for x in range(0, 5):
            if step <= x:
                step_input[x] = 0
            else:
                step_input[4 - x] = normalized_song_data[step - 1 - x][1]

        # Set Future Song Levels
        for x in range(1, 6):
            if step + x >= len(normalized_song_data):
                step_input[7 + x] = 0
            else:
                step_input[7 + x] = normalized_song_data[step + x][1]

        # Set Current Song Level
        step_input[5] = normalized_song_data[step][0]
        step_input[6] = normalized_song_data[step][1]
        step_input[7] = normalized_song_data[step][2]

        step_data.append(step_input)

    print("    Finished Creating Step Data for song")

    return step_data


# ----------------------------------------- YN Model Methods -----------------------------------------------------#

# Builds an RNN for classifying if we should have a note in this frame or not
def build_yn_model():
    print("Creating Neural Network...")

    # Create the training model
    model = tf.keras.Sequential([
        # Input Layer
        tf.keras.layers.Dense(yn_input_size, activation='linear', input_shape=(yn_input_size,)),

        tf.keras.layers.Dense(10, activation='relu', kernel_initializer='random_normal'),

        tf.keras.layers.Dense(yn_output_size, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'],
                  run_eagerly=False)

    return model


# Build the training data for the Y/N network, with the normalized song data, and the transformed note times.
def build_ynn_input(normalized_song_data, song_data):
    print("    Creating Training Data for song")

    step_data = []

    steps = build_step_input(normalized_song_data, yn_input_size)

    for step, step_input in enumerate(steps):
        step_output = [0] * yn_output_size

        # Go through the notes, if the previous step had data, mark it as an input
        for note in song_data:
            if note['_time'] == ((step - 1) * divisions_increment):
                step_input[yn_input_size - 1] = 1
            if note['_time'] == (step * divisions_increment):
                step_output[yn_output_size - 1] = 1

        if step_output[yn_output_size - 1] == 0:
            step_output[yn_output_size - 2] = 1
            dropout_chance = random.randint(1, 101)

            # Drop some zero frames, but leave a buffer
            if step < zero_beat_buffer or step > (len(steps) - zero_beat_buffer) or dropout_chance > yn_zero_value_dropout:
                step_data.append([step_input, step_output])

        else:
            step_data.append([step_input, step_output])

    print("    Training Data for Song Created")
    return step_data


# Build the training data from the target directory
def build_ynn_training_data(directory_path):
    print("Creating Training Data")
    # Go through all the subdirectories
    directories = glob(directory_path + "/*/")

    training_set = []

    for directory in directories:
        song_meta_data, normalized, song_data = get_data_for_directory(directory)

        # Sanity check
        if song_meta_data is not None and normalized is not None:
            for song_map in song_data:
                notes = get_notes_for_song(song_map, song_meta_data['_beatsPerMinute'])
                step_data = build_ynn_input(normalized, notes)
                training_set.extend(step_data)

    training_inputs = []
    training_outputs = []
    for data_pair in training_set:
        training_inputs.append(data_pair[0])
        training_outputs.append(data_pair[1])

    print("Training Data Created")
    return training_inputs, training_outputs


# Get the output from the yn model
def gen_yn_data(normalized, model):
    print("Step 1 of 5: Generating Note Y/N Output")

    results = []
    steps = build_step_input(normalized, yn_input_size)
    yn_notes = 0

    for step, step_input in enumerate(steps):

        step_input = np.array(step_input)
        step_input = np.reshape(step_input, [1, yn_input_size])
        result = model.predict(step_input)

        if step < len(steps) - 1:
            if result[0][yn_output_size - 1] > result[0][yn_output_size - 2]:
                steps[step + 1][yn_input_size - 1] = 1
                yn_notes += 1

        results.append(result[0])

    print('  ', yn_notes, ' note positions generated')
    return results


# ------------------------------------------ RBB Model Methods ---------------------------------------------------#

# Builds an RNN for classifying what color notes we should have (Red, Blue, or Both)
def build_rbb_model():
    print("Creating Neural Network...")

    # Create the training model
    model = tf.keras.Sequential([
        # Input Layer
        tf.keras.layers.Dense(rbb_input_size, activation='linear', input_shape=(rbb_input_size,)),

        tf.keras.layers.Dense(50, activation='relu', kernel_initializer='random_normal'),
        tf.keras.layers.Dense(100, activation='relu', kernel_initializer='random_normal'),
        tf.keras.layers.Dense(50, activation='relu', kernel_initializer='random_normal'),

        tf.keras.layers.Dense(rbb_output_size, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  run_eagerly=False)

    return model


# Build the training data for the Red-Blue-Both Neural Network
def build_rbb_training_data(directory_path):
    print("Creating Training Data")
    # Go through all the subdirectories
    directories = glob(directory_path + "/*/")

    training_set = []

    for directory in directories:
        song_meta_data, normalized, song_data = get_data_for_directory(directory)

        # Sanity check
        if song_meta_data is not None and normalized is not None:
            for song_map in song_data:
                notes = get_notes_for_song(song_map, song_meta_data['_beatsPerMinute'])
                step_data = build_rbb_input(normalized, notes)
                training_set.extend(step_data)

    training_inputs = []
    training_outputs = []
    for data_pair in training_set:
        training_inputs.append(data_pair[0])
        training_outputs.append(data_pair[1])

    print("Training Data Created")
    return training_inputs, training_outputs


# Build the training data for the RBB network, with the normalized song data, and the transformed note times.
def build_rbb_input(normalized_song_data, song_data):
    print("    Creating Training Data for song")

    step_data = []

    steps = build_step_input(normalized_song_data, rbb_input_size)

    for step, step_input in enumerate(steps):
        step_output = [0] * rbb_output_size

        # Go through the notes, if the previous step had data, mark it as an input
        for note in song_data:
            if note['_time'] == ((step - 1) * divisions_increment):
                if note['_type'] == 0:
                    step_input[rbb_input_size - 3] = 1
                if note['_type'] == 1:
                    step_input[rbb_input_size - 2] = 1
            if note['_time'] == (step * divisions_increment):
                if note['_type'] == 0:
                    step_output[rbb_output_size - 3] = 1
                if note['_type'] == 1:
                    step_output[rbb_output_size - 2] = 1

        if step_output[rbb_output_size - 2] == 1 and step_output[rbb_output_size - 3] == 1:
            step_output[rbb_output_size - 1] = 1
            step_output[rbb_output_size - 2] = 0
            step_output[rbb_output_size - 3] = 0

        if step_output[rbb_output_size - 1] == 1:
            # Drop some both frames
            dropout_chance = random.randint(1, 101)
            if dropout_chance > rbb_both_dropout:
                step_data.append([step_input, step_output])

        # Make sure we had SOME kind of output
        elif step_output[rbb_output_size - 2] == 1 or step_output[rbb_output_size - 3] == 1:
            step_data.append([step_input, step_output])

    print("    Training Data for Song Created")
    return step_data


# Get the output from the yn model
def gen_rbb_data(normalized, model, yn_results):
    print("Step 2 of 5: Generating Note Red-Blue-Both Output")

    results = []
    steps = build_step_input(normalized, rbb_input_size)
    red_notes = 0
    blue_notes = 0
    both_notes = 0

    for step, step_input in enumerate(steps):
        if yn_results[step][yn_output_size - 1] > yn_results[step][yn_output_size - 2]:

            step_input = np.array(step_input)
            step_input = np.reshape(step_input, [1, rbb_input_size])
            result = model.predict(step_input)

            if step < len(steps) - 1:
                highest_value = 0
                highest_index = 0
                for output_value in range(rbb_output_size):
                    if result[0][output_value] > highest_value:
                        highest_value = result[0][output_value]
                        highest_index = output_value

                steps[step + 1][rbb_input_size - 1 - highest_index] = 1
                if highest_index == 0:
                    red_notes += 1
                if highest_index == 1:
                    blue_notes += 1
                if highest_index == 2:
                    both_notes += 1

            results.append([result[0], step])

    print('   Total Notes: ', (red_notes + blue_notes + (both_notes * 2)))
    print('     Red Notes: ', red_notes)
    print('     Blue Notes: ', blue_notes)
    print('     Both Notes: ', both_notes)
    return results


# -------------------------------------------- Util Methods ------------------------------------------------------#


# Round to the nearest divisions_increment
def x_round(x):
    return math.ceil(x * division_factor) / division_factor


# Build a note map and return it.
def build_note(step, layer, index, cut_direction, color):
    new_note = {'_time': step * divisions_increment,
                '_lineIndex': index,
                '_lineLayer': layer,
                '_cutDirection': cut_direction,
                '_type': color}

    return new_note


# Build the notes for the song
def build_yn_song_output(results):
    notes = []
    cut_r = 1
    cut_b = 1
    color = 1
    for step, result in enumerate(results):
        if result[1] > result[0]:
            if color == 1:
                notes.append(build_note(step, 0, 2, cut_b % 2, 1))
                cut_b = (cut_b + 1) % 2
                color = 0
            else:
                notes.append(build_note(step, 0, 1, cut_r % 2, 0))
                cut_r = (cut_r + 1) % 2
                color = 1

    return notes


# Build the notes for the song
def build_rbb_song_output(results):
    notes = []
    cut_r = 1
    cut_b = 1

    for step, result in enumerate(results):
        rbb_model_output = result[0]
        time_step = result[1]
        highest_value = 0
        highest_index = 0
        for output_value in range(rbb_output_size):
            if rbb_model_output[output_value] > highest_value:
                highest_value = rbb_model_output[output_value]
                highest_index = output_value

        if highest_index == 0 or highest_index == 2:
            notes.append(build_note(time_step, 0, 2, cut_b % 2, 1))
            cut_b = (cut_b + 1) % 2

        if highest_index == 1 or highest_index == 2:
            notes.append(build_note(time_step, 0, 1, cut_r % 2, 0))
            cut_r = (cut_r + 1) % 2

    return notes


# Writes the notes out to a file
def write_output_file(output_file, notes):
    output_dict = {
        "_version": "1.0.0",
        "_BPMChanges": [],
        "_events": [],
        "_notes": notes,
        "_obstacles": [],
        "_bookmarks": []
    }

    f = open(output_file + ".notes.json", 'w')
    json.dump(output_dict, f)
    f.close()


# Main=
def main():
    global bpm, division_factor, divisions_increment, difficulty

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Generate a Beat Saber Song from a .ogg/.egg file.')
    parser.add_argument('input', action='store', type=str, help='Input file')
    parser.add_argument('directory', action='store', type=str, help='Directory for Training')
    parser.add_argument('bpm', action='store', type=int, help='Beats Per Minute', default=200)
    parser.add_argument('divisions', action='store', type=float, help='Divisions Per Beat', default=2.0)
    parser.add_argument('difficulty', action='store', type=str, help='Target Difficulty', default='Expert')
    args = parser.parse_args()

    bpm = args.bpm
    division_factor = args.divisions
    divisions_increment = round(1.0 / division_factor, 2)
    difficulty = args.difficulty
    
    # Load the data for out target song
    target_song_normalized = normalize_audio_file(args.input)

    # Yes/No Note Training

    if not regen_nn_models and path.exists(os.path.join('./models/' + difficulty + '-yn.h5')):
        yn_model = tf.keras.models.load_model('./models/' + difficulty + '-yn.h5')

    else:
        yn_model = build_yn_model()
        training_inputs, training_outputs = build_ynn_training_data(args.directory)

        yn_model.fit(training_inputs, training_outputs, epochs=yn_training_iterations)
        yn_model.save('./models/' + difficulty + '-yn.h5')

    # Red/Blue Note Training

    if not regen_nn_models and path.exists(os.path.join('./models/' + difficulty + '-rbb.h5')):
        rbb_model = tf.keras.models.load_model('./models/' + difficulty + '-rbb.h5')
    else:
        rbb_model = build_rbb_model()
        training_inputs, training_outputs = build_rbb_training_data(args.directory)

        rbb_model.fit(training_inputs, training_outputs, epochs=rbb_training_iterations)
        rbb_model.save('./models/' + difficulty + '-rbb.h5')

    print("Creating new song...")
    yn_results = gen_yn_data(target_song_normalized, yn_model)
    rbb_results = gen_rbb_data(target_song_normalized, rbb_model, yn_results)
    notes = build_rbb_song_output(rbb_results)
    write_output_file(args.input, notes)


main()
