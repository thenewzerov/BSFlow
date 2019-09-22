import json
import numpy as np
import math
import os
import sys
import soundfile as sf


class Utils:

    @staticmethod
    def normalize_audio_file(file_path, bpm, notes_per_beat, regen_files=False):
        """
        Normalize an audio file for training or generating
        :param file_path: Path to the audio file
        :param bpm: Beats Per Minute to read the song in at
        :param notes_per_beat: Max number of notes per beat
        :param regen_files: Should the file be forced to be re-created
        :return: The Normalized audio file
        """
        print("    Normalizing ", file_path)

        with sf.SoundFile(file_path, 'r') as f:
            data = f.read()

            beats_in_song = round(((len(f) / f.samplerate) / 60) * bpm)
            frames_per_step = round(len(f) / (beats_in_song * notes_per_beat))
            print('    Samples         = {}'.format(len(f)))
            print('    Sample Rate     = {}'.format(f.samplerate))
            print('    Seconds         = {}'.format(len(f) / f.samplerate))
            print('    Beats In  Song  = {} (at {} BPM)'.format(beats_in_song, bpm))
            print('    Frames Per Step = {}'.format(frames_per_step))
            print('    Total Steps     = {}'.format(beats_in_song * notes_per_beat))

            # Create an array to store the normalized data
            normalized = []

            beat_sum = 0
            frame_count = 0
            min_frame = sys.maxsize
            max_frame = 0

            if os.path.exists(os.path.join(file_path + '-' + str(bpm) + '-' + str(notes_per_beat) + '-.json')) and not regen_files:
                with open(os.path.join(file_path + '-' + str(bpm) + '-' + str(notes_per_beat) + '.json')) as json_file:
                    normalized = json.load(json_file)
                    json_file.close()
                    print("    Loaded normalized file. ", len(normalized), " steps created.")
                    return np.array(normalized)

            for x in range(len(data)):
                for c in range(f.channels):
                    if data[x][c] < min_frame:
                        min_frame = data[x][c]
                    if data[x][c] > max_frame:
                        max_frame = data[x][c]

                    beat_sum += data[x][c]

                frame_count += 1

                if frame_count == frames_per_step:
                    average = (beat_sum / frames_per_step)
                    normalized.append([min_frame, average, max_frame])
                    beat_sum = 0
                    frame_count = 0
                    min_frame = sys.maxsize
                    max_frame = 0

            normalized = np.array(normalized)
            normalized = normalized - normalized.mean()
            normalized = normalized / np.abs(normalized).max()

            print('   Total Beats = {}'.format(len(normalized)))

        output_array = normalized.tolist()
        f = open(file_path + '-' + str(bpm) + '-' + str(notes_per_beat) + ".json", 'w')
        json.dump(output_array, f)
        f.close()

        print("    Normalizing complete. ", len(normalized), " steps created.")
        return normalized

    @staticmethod
    def x_round(x, notes_per_beat):
        """
        Rounds to the nearest note increment
        :param x: number to round
        :param notes_per_beat: how many notes per beat
        :return:
        """
        return math.ceil(x * notes_per_beat) / notes_per_beat

    @staticmethod
    def get_notes_for_song(beat_saber_song_dat, original_bpm, bpm, accuracy):
        """
        Build the song data for a normalized song and a beat saber .dat file.  Normalizes the note times
        :param beat_saber_song_dat: Song's info.dat file read is as JSON
        :param original_bpm: Original BPM for the song
        :param bpm: Target BPM to convert the note times to
        :param accuracy: How accurate we should calculate the song levels at
        :return: JSON array with the notes
        """
        print("Calculating Note Times")

        converted_notes = []

        if '_notes' in beat_saber_song_dat:

            notes = beat_saber_song_dat['_notes']
            notes = sorted(notes, key=lambda nt: nt['_time'])

            for note in notes:
                if note['_type'] == 0 or note['_type'] == 1:
                    note['_time'] = Utils.x_round((note['_time'] / original_bpm) * bpm, accuracy)
                    converted_notes.append(note)

        print("Notes converted")

        return converted_notes

    @staticmethod
    def get_data_for_directory(song_directory, difficulty, bpm, notes_per_beat):
        """
        Returns all the song data for a directory
        :param song_directory: The directory containing the song to load
        :param difficulty: Difficulty to pull info for
        :return: Song data as JSON
        """
        normalized = None
        song_meta_data = None
        song_data = []

        print(" Building Training Data for ", song_directory)

        # Normalize the song data
        for file in os.listdir(song_directory):
            if file.endswith(".ogg") or file.endswith(".egg"):
                normalized = Utils.normalize_audio_file(os.path.join(song_directory, file), bpm, notes_per_beat)

        # Might as well do a sanity check to make sure we found a sound file
        if normalized is not None:

            # Load the original Song metadata so we can get the beats per minute
            for file in os.listdir(song_directory):
                if file.endswith(".dat") and file.startswith('info'):
                    with open(os.path.join(song_directory, file)) as info_file:
                        song_meta_data = json.load(info_file)

            # Make sure we got metadata!
            if song_meta_data is not None:
                for file in os.listdir(song_directory):
                    if file.endswith(difficulty + ".dat") and not file.startswith('info'):
                        with open(os.path.join(song_directory, file)) as json_file:
                            beat_saber_song_dat = json.load(json_file)
                            song_data.append(beat_saber_song_dat)

        print(' ', song_directory, " completed")
        return song_meta_data, normalized, song_data

    @staticmethod
    def build_step_input(normalized_song_data, total_size):
        """
        Build the step input data, last values will not be filled out
        :param normalized_song_data: The Normalized song data
        :param total_size: Total size of each output arrays
        :return: Array of the normalized song data
        """
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

    @staticmethod
    def build_note(step, layer, index, cut_direction, color, notes_per_beat):
        """
        Builds a note
        :param step: Step Number (will be converted to a time)
        :param layer: :The line layer
        :param index: The line index
        :param cut_direction: The cut direction
        :param color: The note color
        :param notes_per_beat: The number of notes per beat
        :return: The completed note
        """
        new_note = {'_time': step,
                    '_lineIndex': index,
                    '_lineLayer': layer,
                    '_cutDirection': cut_direction,
                    '_type': color}

        return new_note

    @staticmethod
    def set_note_times(notes, notes_per_beat):
        """

        :param notes:
        :param notes_per_beat:
        :return:
        """
        for note in notes:
            note["_time"] = note["_time"] / notes_per_beat

        return notes

    @staticmethod
    def write_output_file(output_file, notes):
        """
        Writes the song data out in the Beat Saber song format
        :param output_file: The path for the data file
        :param notes: The JSON array of notes
        :return: Nothing
        """
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
