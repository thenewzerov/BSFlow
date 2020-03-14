from networks.BeatSaberNetwork import BeatSaberNetwork
import numpy as np
import random
import tensorflow as tf
from utils.Utils import Utils


class RBBNetwork(BeatSaberNetwork):

    def set_sizes(self):
        self.input_size = 16
        self.output_size = 3

    def build_model(self):
        """
        Builds the Yes/No Model
        :return:
        """
        print("Creating RBB Neural Network...")

        # Create the training model
        self.model = tf.keras.Sequential([
            # Input Layer
            tf.keras.layers.Dense(self.input_size, activation='linear', input_shape=(self.input_size,)),

            tf.keras.layers.Dense(50, activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(100, activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(100, activation='softmax', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(100, activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(50, activation='relu', kernel_initializer='random_normal'),

            tf.keras.layers.Dense(self.output_size, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=False)

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
                step_data = self.build_rbb_input(normalized, notes)
                training_set.extend(step_data)

        random.shuffle(training_set)

        training_inputs = []
        training_outputs = []
        for data_pair in training_set:
            training_inputs.append(data_pair[0])
            training_outputs.append(data_pair[1])

        print("Training Data Created")
        return training_inputs, training_outputs

    def build_rbb_input(self, normalized_song_data, song_data):
        """

        :param normalized_song_data:
        :param song_data:
        :return:
        """
        print("    Creating Training Data for song")

        step_data = []

        steps = Utils.build_step_input(normalized_song_data, self.input_size)

        for step, step_input in enumerate(steps):
            step_output = [0] * self.output_size

            # Go through the notes, if the previous step had data, mark it as an input
            for note in song_data:
                if note['_time'] == ((step - 1) * self.notes_per_beat):
                    if note['_type'] == 0:
                        step_input[self.input_size - 3] = 1
                    if note['_type'] == 1:
                        step_input[self.input_size - 2] = 1
                if note['_time'] == (step * self.notes_per_beat):
                    if note['_type'] == 0:
                        step_output[self.output_size - 3] = 1
                    if note['_type'] == 1:
                        step_output[self.output_size - 2] = 1

            if step_output[self.output_size - 2] == 1 and step_output[self.output_size - 3] == 1:
                step_output[self.output_size - 1] = 1
                step_output[self.output_size - 2] = 0
                step_output[self.output_size - 3] = 0

            step_data.append([step_input, step_output])

        print("    Training Data for Song Created")
        return step_data

    def gen_data(self, normalized, yn_gen_results, yn_output_size):
        """
        Generate the data from the RBB Model
        :param normalized: Normalized song data
        :param yn_gen_results: Inputs from a YesNoModel
        :param yn_output_size: The output size of each YesNo Model
        :return: The song data
        """
        print("Step 2 of 5: Generating Note Red-Blue-Both Output")

        results = []
        steps = Utils.build_step_input(normalized, self.input_size)
        red_notes = 0
        blue_notes = 0
        both_notes = 0
        no_notes = 0

        for step, step_input in enumerate(steps):
            if yn_gen_results[step] == 1:

                step_input = np.array(step_input)
                step_input = np.reshape(step_input, [1, self.input_size])
                result = self.model.predict(step_input)

                if step < len(steps) - 1:
                    highest_value = 0
                    highest_index = 0
                    for output_value in range(self.output_size):
                        if result[0][output_value] > highest_value:
                            highest_value = result[0][output_value]
                            highest_index = output_value

                    steps[step + 1][self.input_size - 1 - highest_index] = 1
                    if highest_index == 0:
                        red_notes += 1
                    if highest_index == 1:
                        blue_notes += 1
                    if highest_index == 2:
                        both_notes += 1

                results.append([result[0], step])

            else:
                no_notes += 1

        print('   Total Notes : ', (red_notes + blue_notes + (both_notes * 2)))
        print('     Red Notes : ', red_notes)
        print('     Blue Notes: ', blue_notes)
        print('     Both Notes: ', both_notes)
        print('     No Notes  : ', no_notes)
        return results
