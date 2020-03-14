import numpy as np
from utils.Utils import Utils


def high_pass_filter(normalized, lower_bound):
    results = []
    total_notes = 0
    for frame in normalized:
        if frame[2] > lower_bound:
            results.append(1)
            total_notes = total_notes + 1
        else:
            results.append(0)
    return results, total_notes


class YesNoNetwork:

    def __init__(self):
        self.input_size = 10
        self.output_size = 10
        self.set_sizes()

    def set_sizes(self):
        self.input_size = 14
        self.output_size = 5

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

        lower_bound = 1.5
        results, total_notes = high_pass_filter(normalized, lower_bound)

        while total_notes < (len(normalized) * .8):
            lower_bound = lower_bound - abs(lower_bound / 100)
            results, total_notes = high_pass_filter(normalized, lower_bound)

        print('  ', total_notes, ' note positions generated')
        print('  ', (len(normalized) - total_notes), ' position with no notes generated')

        final_data = []
        for index, x in enumerate(results):
            frame = [0, 0, x, 0, 0]

            if index > 1:
                frame[0] = results[index - 2]
            if index > 0:
                frame[1] = results[index - 1]
            if index < len(results)-1:
                frame[3] = results[index + 1]
            if index < len(results)-2:
                frame[4] = results[index + 2]

            final_data.append(frame)

        return results
