import argparse

from networks.LineIndexNetwork import LineIndexNetwork
from networks.LineLayerNetwork import LineLayerNetwork
from networks.RBBNetwork import RBBNetwork
from networks.YesNoNetwork import YesNoNetwork
from utils.Utils import Utils


# Build the notes for the song
def build_rbb_song_output(results, rbb_output_size, notes_per_beat):
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
            notes.append(Utils.build_note(time_step, 0, 2, cut_b % 2, 1, notes_per_beat))
            cut_b = (cut_b + 1) % 2

        if highest_index == 1 or highest_index == 2:
            notes.append(Utils.build_note(time_step, 0, 1, cut_r % 2, 0, notes_per_beat))
            cut_r = (cut_r + 1) % 2

    return notes


# Get the arguments and set any global variables
def get_args():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Generate a Beat Saber Song from a .ogg/.egg file.')
    parser.add_argument('input', action='store', type=str, help='Audio input file (.ogg or .egg)')
    parser.add_argument('directory', action='store', type=str, help='Directory containing directories of Beat Saber songs to train the networks with')
    parser.add_argument('bpm', action='store', type=int, help='Beats Per Minute for the final song and the training', default=200)
    parser.add_argument('notes_per_beat', action='store', type=float, help='Maximum numbner of notes per beat', default=2.0)
    parser.add_argument('difficulty', action='store', type=str, help='Target Difficulty for the final song', default='Expert')
    parser.add_argument('iterations', action='store', type=int, help='Training Iterations for the networks', default=5000)
    parser.add_argument('--force-regen', action='store_true', help='Force the models and song data to be recreated.')
    args = parser.parse_args()

    return args


# Main
def main():

    args = get_args()

    # Load the models
    yn_model = YesNoNetwork('YNNetwork', args.difficulty, args.bpm, args.directory, args.iterations, 2, args.notes_per_beat, regen_model=args.force_regen)
    rbb_model = RBBNetwork('RBBNetwork', args.difficulty, args.bpm, args.directory, args.iterations, 2, args.notes_per_beat, regen_model=args.force_regen)
    li_model = LineIndexNetwork('LINetwork', args.difficulty, args.bpm, args.directory, args.iterations, 2, args.notes_per_beat, regen_model=args.force_regen)
    ll_model = LineLayerNetwork('LLNetwork', args.difficulty, args.bpm, args.directory, args.iterations, 2,args.notes_per_beat, regen_model=args.force_regen)

    # Load the data for out target song
    target_song_normalized = Utils.normalize_audio_file(args.input, args.bpm, args.notes_per_beat)

    print("\nCreating new song...\n")
    yn_results = yn_model.gen_data(target_song_normalized)
    rbb_results = rbb_model.gen_data(target_song_normalized, yn_results, yn_model.output_size)
    notes = build_rbb_song_output(rbb_results, rbb_model.output_size, args.notes_per_beat)
    li_results, notes = li_model.gen_data(target_song_normalized, notes)
    ll_results, notes = ll_model.gen_data(target_song_normalized, notes)

    print("\nFinalizing Song...\n")
    notes = Utils.set_note_times(notes, args.notes_per_beat)
    Utils.write_output_file(args.input, notes)


main()
