import numpy as np
import itertools
from itertools import groupby
from pprint import pprint


class SnakeCubeSolver(object):
    _all_axes = ('+x', '-x', '+y', '-y', '+z', '-z')
    _axis_location_delta_map = {
        '+x': (1, 0, 0),
        '-x': (-1, 0, 0),
        '+y': (0, 1, 0),
        '-y': (0, -1, 0),
        '+z': (0, 0, 1),
        '-z': (0, 0, -1)
    }
    _orthogonal_axis_map = {
        '+x': ('+y', '-y', '+z', '-z'),
        '-x': ('+y', '-y', '+z', '-z'),
        '+y': ('+x', '-x', '+z', '-z'),
        '-y': ('+x', '-x', '+z', '-z'),
        '+z': ('+x', '-x', '+y', '-y'),
        '-z': ('+x', '-x', '+y', '-y')
    }

    def __init__(self, dim):
        self._dim = int(dim)
        self._num_cells = self._dim ** 3
        self._segment_lengths = None
        self._is_joint_cell = None

    def input_segment_lengths(self, segment_lengths):
        self._segment_lengths = segment_lengths
        self._is_joint_cell = self._segment_lengths_to_is_joint_cell(segment_lengths)

    def solve(self):
        # sanity check
        assert self._segment_lengths is not None
        assert self._is_joint_cell is not None
        assert np.sum(segment_lengths) == self._num_cells

        # map of the filled cells
        filled_cells = np.zeros((self._dim, self._dim, self._dim), dtype=bool)

        # cell_locations[i] = [x, y, z], location for the i-th cell
        cell_locations = [None] * self._num_cells

        # cell_axes[i] is the vector direction from cells[i] to cells[i+1]
        cell_axes = [None] * (self._num_cells - 1)

        return self._solve(0, filled_cells, cell_locations, cell_axes)

    def _segment_lengths_to_is_joint_cell(self, segment_lengths):
        # determine middle cells
        # that is, the cell's input and output axes are parallel
        is_joint_cell = np.zeros((self._num_cells,), dtype=bool)
        segment_lengths_cumsum = np.cumsum(segment_lengths)
        for segment_index, segment_length in enumerate(segment_lengths):
            # get start and end index of joints
            end_cell_index = segment_lengths_cumsum[segment_index] - 1
            start_cell_index = end_cell_index - segment_length + 1

            # if length >= 2, then start and end are joints
            if end_cell_index - start_cell_index >= 1:
                is_joint_cell[start_cell_index] = True
                is_joint_cell[end_cell_index] = True

        return is_joint_cell

    def _get_location(self, prev_location, prev_axis, filled_cells):
        x, y, z = prev_location
        dx, dy, dz = SnakeCubeSolver._axis_location_delta_map[prev_axis]
        location = x + dx, y + dy, z + dz
        for val in location:
            if val >= self._dim or val < 0:
                return None
        if filled_cells[location] == True:
            return None
        return location

    def _solve(self, index, filled_cells, cell_locations, cell_axes):
        # base case
        if index == 0:
            half_dim = int((self._dim + 1) / 2)
            start_index_range = [list(range(half_dim))]
            for init_location in itertools.product(*(start_index_range * 3)):
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

        # if index == self._num_cells - 1, then done
        if index == self._num_cells - 1:
            return (cell_locations, cell_axes)

        if self._is_joint_cell[index]:
            axes = SnakeCubeSolver._orthogonal_axis_map[cell_axes[index - 1]]
        else:
            axes = [cell_axes[index - 1]]
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

    solver = SnakeCubeSolver(4)
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
