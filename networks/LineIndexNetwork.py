from networks.BeatSaberNetwork import BeatSaberNetwork
import numpy as np
import random
import tensorflow as tf
from utils.Utils import Utils


class LineIndexNetwork(BeatSaberNetwork):

    def set_sizes(self):
        self.input_size = 19
        self.output_size = 4

    def build_model(self):
        """
        Builds the Yes/No Model
        :return:
        """
        print("Creating LineIndex Neural Network...")

        self.input_size = 19
        self.output_size = 4

        # Create the training model
        self.model = tf.keras.Sequential([
            # Input Layer
            tf.keras.layers.Dense(self.input_size, activation='linear', input_shape=(self.input_size,)),

            tf.keras.layers.Dense(50, activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(100, activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(50, activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(100, activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(50, activation='relu', kernel_initializer='random_normal'),

            tf.keras.layers.Dense(self.output_size, activation='softmax')
        ])

        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=False)

    def build_training_data(self, directory):
        """
        Builds Training data for a song directory
        :param directory: Song directory
        :return: Training data for a song
        """

        training_set = []

        song_meta_data, normalized, song_data = Utils.get_data_for_directory(directory, self.difficulty, self.bpm, self.notes_per_beat)

        # Sanity check
        if song_meta_data is not None and normalized is not None:
            for song_map in song_data:
                notes = Utils.get_notes_for_song(song_map, song_meta_data['_beatsPerMinute'], self.bpm, self.accuracy)
                step_data = self.build_li_input(normalized, notes)
                training_set.extend(step_data)

        random.shuffle(training_set)

        training_inputs = []
        training_outputs = []
        for data_pair in training_set:
            training_inputs.append(data_pair[0])
            training_outputs.append(data_pair[1])

        print("Training Data Created")
        return training_inputs, training_outputs

    def build_li_input(self, normalized_song_data, song_data):

        max_note_look_back = 3

        """
        Build the input data.
        :param normalized_song_data:
        :param song_data:
        :return:
        """
        print("    Creating Training Data for song")

        step_data = []

        steps = Utils.build_step_input(normalized_song_data, self.input_size)

        for step, step_input in enumerate(steps):
            step_output = [0] * self.output_size

            note_input_found = False
            # Go through the notes, if the previous step had data, mark it as an input
            for note_index, note in enumerate(song_data):
                if note['_time'] == (step * self.notes_per_beat):
                    step_output[note['_lineIndex']] = 1
                    note_input_found = True

                    last_matching_note = None
                    for previous_notes_index in range(note_index - max_note_look_back, note_index):
                        previous_note = song_data[previous_notes_index]
                        if previous_note['_type'] == note['_type']:
                            last_matching_note = previous_note

                    if last_matching_note is not None:
                        step_input[last_matching_note['_lineIndex'] + 13] = 1

                    step_input[self.input_size - 1 - note['_type']] = 1

            if note_input_found:
                step_data.append([step_input, step_output])

        print("    Training Data for Song Created")
        return step_data

    def gen_data(self, normalized, notes):
        """
        Generate the data from the RBB Model
        :param normalized: Normalized song data
        :param notes: Notes to create data for
        :return: The song data
        """
        print("Step 3 of 5: Generating Note Line Index Output")

        max_note_look_back = 3

        results = []
        steps = Utils.build_step_input(normalized, self.input_size)
        line_index_counts = [0] * self.output_size

        for note_index, note in enumerate(notes):
            step_index = int(note['_time'])
            step = steps[step_index]

            lowest_look_back_time = (step_index - max_note_look_back)
            if lowest_look_back_time < 0:
                lowest_look_back_time = 0

            matching_positions = [0] * self.output_size
            last_matching_note = None
            for previous_notes_index in range(note_index - max_note_look_back, note_index):
                if previous_notes_index >= 0:
                    previous_note = notes[previous_notes_index]
                    if previous_note['_type'] == note['_type'] and previous_note['_time'] > lowest_look_back_time:
                        last_matching_note = previous_note
                    if previous_note['_time'] == note['_time']:
                        matching_positions[previous_note['_lineIndex']] = 1

            if last_matching_note is not None:
                step[last_matching_note['_lineIndex'] + 13] = 1

            step[self.input_size - 1 - note['_type']] = 1

            step_input = np.array(step)
            step_input = np.reshape(step_input, [1, self.input_size])

            result = self.model.predict(step_input)
            highest_value = -1
            highest_index = 0
            for x in range(0, self.output_size):
                if result[0][x] > highest_value and matching_positions[x] != 1:
                    highest_index = x
                    highest_value = result[0][x]

            note['_lineIndex'] = highest_index
            line_index_counts[highest_index] += 1

            results.append(result[0])

        print('   Totals Per Line Index: ', line_index_counts)
        return results, notes
