import json
from abc import abstractmethod
from glob import glob
import os
import tensorflow as tf


class BeatSaberNetwork:

    def __init__(self, name, difficulty, bpm, training_directory, training_iterations,
                 accuracy, notes_per_beat, zero_beat_buffer=15, regen_model=False):
        """
        Creates a new BeatSaverNetwork
        :param difficulty: Difficulty this is trained for
        :param bpm: BPM for the model
        :param regen_model: Should the model be force created. Defaults to false
        """
        self.difficulty = difficulty
        self.bpm = bpm
        self.input_size = 10
        self.output_size = 10
        self.set_sizes()
        self.accuracy = accuracy
        self.notes_per_beat = notes_per_beat
        self.zero_beat_buffer = zero_beat_buffer
        self.name = name

        if not regen_model and os.path.exists(os.path.join('./models/' + self.name + '-' + self.difficulty + '-' + str(self.bpm) + '-' + str(self.notes_per_beat) + '-yn.h5')):
            self.model = tf.keras.models.load_model('./models/' + self.name + '-' + self.difficulty + '-' + str(self.bpm) + '-' + str(self.notes_per_beat) + '-yn.h5')

        else:
            self.build_model()
            self.train_model(training_directory, training_iterations, notes_per_beat)

    def train_model(self, directory, training_iterations, notes_per_beat):

        directories = glob(directory + "/*/")

        for directory in directories:
            print("   Loading data from ", directory)
            path = (directory + self.name + "-" + self.difficulty + "-" + str(self.bpm) + "-" + str(self.notes_per_beat) + ".json")

            if os.path.exists(path):
                with open(path) as json_file:
                    training_data = json.load(json_file)
                    training_inputs = training_data["training_inputs"]
                    training_outputs = training_data["training_outputs"]
            else:
                training_inputs, training_outputs = self.build_training_data(directory)
                training_data = {"training_inputs": training_inputs, "training_outputs": training_outputs}
                f = open(path, 'w')
                json.dump(training_data, f)

            if len(training_inputs) > 0 and len(training_outputs) > 0:
                self.model.fit(training_inputs, training_outputs, epochs=training_iterations)
        self.model.save('./models/' + self.name + '-' + self.difficulty + '-' + str(self.bpm) + '-' + str(notes_per_beat) + '-yn.h5')

    @abstractmethod
    def build_training_data(self, directory):
        pass

    @abstractmethod
    def set_sizes(self):
        pass

    @abstractmethod
    def build_model(self):
        pass
