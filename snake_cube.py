import numpy as np
import itertools
from itertools import groupby
from pprint import pprint


class SnakeCubeSolver(object):
    _all_axes = ['+x', '-x', '+y', '-y', '+z', '-z']
    _map_axis_location_delta = {
        '+x': (1, 0, 0),
        '-x': (-1, 0, 0),
        '+y': (0, 1, 0),
        '-y': (0, -1, 0),
        '+z': (0, 0, 1),
        '-z': (0, 0, -1)
    }

    def __init__(self):
        self.segment_lengths = None
        self.is_middle_cell = None

    def input_segment_lengths(self, segment_lengths):
        self.segment_lengths = segment_lengths
        self.is_middle_cell = self._segment_lengths_to_middle_cell()

    def solve(self):
        # map of the filled cells
        filled_cells = np.zeros((4, 4, 4), dtype=bool)

        # cells[i] = [x, y, z], location for the i-th cell
        cell_locations = [None] * 64

        # axes[i] is the vector direction from cells[i] to cells[i+1]
        cell_axes = [None] * 63

        return self._solve(0, filled_cells, cell_locations, cell_axes)

    def _segment_lengths_to_middle_cell(self):
        # determine middle cells
        # that is, the cell's input and output axes are parallel
        is_middle_cell = np.zeros((64,), dtype=bool)
        segment_lengths_cumsum = np.cumsum(segment_lengths)
        for segment_index, segment_length in enumerate(segment_lengths):
            start_cell_index = segment_lengths_cumsum[
                segment_index - 1] if segment_index > 0 else 0
            end_cell_index = start_cell_index + segment_length - 1
            if start_cell_index == end_cell_index:
                # case 1: length is 1
                is_middle_cell[start_cell_index] = True
            else:
                # case 2: length is n > 1, and the cell is not the 0th or the (n-1)th
                for i in range(start_cell_index + 1, end_cell_index):
                    is_middle_cell[i] = True
        return is_middle_cell

    def _get_location(self, prev_location, prev_axis, filled_cells):
        x, y, z = prev_location
        dx, dy, dz = SnakeCubeSolver._map_axis_location_delta[prev_axis]
        location = x + dx, y + dy, z + dz
        for val in location:
            if val >= 4 or val < 0:
                return None
        if filled_cells[location] == True:
            return None
        return location

    def _get_axes(self, cell_index, prev_axis):
        assert prev_axis in SnakeCubeSolver._all_axes
        if self.is_middle_cell[cell_index]:
            return [prev_axis]
        if prev_axis in {'+x', '-x'}:
            return ['+y', '-y', '+z', '-z']
        if prev_axis in {'+y', '-y'}:
            return ['+x', '-x', '+z', '-z']
        if prev_axis in {'+z', '-z'}:
            return ['+x', '-x', '+y', '-y']

    def _solve(self, index, filled_cells, cell_locations, cell_axes):
        # base case
        if index == 0:
            for init_location in itertools.product(*([list(range(2))] * 3)):
                print("init location", init_location)
                for init_axis in SnakeCubeSolver._all_axes:
                    # fill the 0-th cell and axis
                    cell_locations[0] = init_location
                    cell_axes[0] = init_axis
                    filled_cells[cell_locations[0]] = True
                    # solve
                    res = self._solve(1, filled_cells, cell_locations,
                                      cell_axes)
                    if res is not None:
                        return res
                    # clean up
                    filled_cells[cell_locations[0]] = False
                    cell_locations[0] = None
                    cell_axes[0] = None
            return None

        # get location
        location = self._get_location(cell_locations[index - 1],
                                      cell_axes[index - 1],
                                      filled_cells)
        if location is None:
            return None

        # fill
        cell_locations[index] = location
        filled_cells[location] = True

        # done if index == 63, cell_locations, cell_axes fully filled
        if index == 63:
            return (cell_locations, cell_axes)

        # get axes
        axes = self._get_axes(index, cell_axes[index - 1])
        for axis in axes:
            # fill axis
            cell_axes[index] = axis
            # try solving recursively
            res = self._solve(index + 1, filled_cells, cell_locations,
                              cell_axes)
            if res is not None:
                return res
            # clean up
            cell_axes[index] = None

        # not successful, clean up
        cell_locations[index] = None
        filled_cells[location] = False

        return None


if __name__ == '__main__':
    # input data
    segment_lengths = np.array(
        [3, 3, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 3, 1, 2, 2, 2, 1, 4, 2, 4,
         3, 2, 2, 2, 2, 4, 1], dtype=int)

    solver = SnakeCubeSolver()
    solver.input_segment_lengths(segment_lengths)
    res = solver.solve()

    if res is not None:
        cell_locations, cell_axes = res

        translate = {
            '+x': 'front',
            '-x': 'back',
            '+y': 'right',
            '-y': 'left',
            '+z': 'up',
            '-z': 'down'
        }
        grouped_cell_axes = [(k, sum(1 for i in g)) for k, g in
                             groupby(cell_axes)]
        grouped_cell_axes = [(translate[x], y + 1) for (x, y) in
                             grouped_cell_axes]

        pprint(grouped_cell_axes)

        lengths = [y for (x, y) in grouped_cell_axes]
        total_lengths = np.sum(lengths) - (len(lengths) - 1)
        print(total_lengths)