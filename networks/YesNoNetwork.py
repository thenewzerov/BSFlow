from networks.BeatSaberNetwork import BeatSaberNetwork
import numpy as np
import random
import tensorflow as tf
from utils.Utils import Utils


class YesNoNetwork(BeatSaberNetwork):

    def set_sizes(self):
        self.input_size = 14
        self.output_size = 2

    def build_model(self):
        """
        Builds the Yes/No Model
        :return:
        """
        print("Creating YNN Neural Network...")

        # Create the training model
        model = tf.keras.Sequential([
            # Input Layer
            tf.keras.layers.Dense(self.input_size, activation='linear', input_shape=(self.input_size,)),

            tf.keras.layers.Dense(40, activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(80, activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(100, activation='softmax', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(50, activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(40, activation='softmax', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(20, activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(10, activation='relu', kernel_initializer='random_normal'),

            tf.keras.layers.Dense(self.output_size, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'],
                      run_eagerly=False)

        self.model = model

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
                step_data = self.build_ynn_input(normalized, notes)
                training_set.extend(step_data)

        random.shuffle(training_set)

        training_inputs = []
        training_outputs = []
        for data_pair in training_set:
            training_inputs.append(data_pair[0])
            training_outputs.append(data_pair[1])

        print("Training Data Created")
        return training_inputs, training_outputs

    def build_ynn_input(self, normalized_song_data, song_data):
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
                    step_input[self.input_size - 1] = 1
                if note['_time'] == (step * self.notes_per_beat):
                    step_output[self.output_size - 1] = 1

            step_data.append([step_input, step_output])

        print("    Training Data for Song Created")
        return step_data

    def gen_data(self, normalized):
        """
        Generate the data from the Yes No Model
        :param normalized: Normalized song data
        :return: The song data
        """
        print("Step 1 of 5: Generating Note Y/N Output")

        results = []
        steps = Utils.build_step_input(normalized, self.input_size)
        y_notes = 0
        n_notes = 0

        for step, step_input in enumerate(steps):

            step_input = np.array(step_input)
            step_input = np.reshape(step_input, [1, self.input_size])
            result = self.model.predict(step_input)

            if step < len(steps) - 1:
                if result[0][self.output_size - 1] > result[0][self.output_size - 2]:
                    steps[step + 1][self.input_size - 1] = 1
                    y_notes += 1
                else:
                    n_notes += 1

            results.append(result[0])

        print('  ', y_notes, ' note positions generated')
        print('  ', n_notes, ' position with no notes generated')
        return results
