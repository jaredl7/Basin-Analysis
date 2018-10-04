import numpy as np
from copy import copy
from itertools import product
from collections import OrderedDict, namedtuple


# These are the functions are allowed to be exported. (The others are used internally, and offer no
# tangible benefit external to this solution.)
__all__ = 'tiled_grid,coarse_grain_hist,determine_histogram_directions,partition'.split(',')


# A lightweight read-only data structure to save and access basin information for a given
# simulation.
Basin = namedtuple('Basin', 'center indices area relative_area weight')


# Define the directions for a 3 x 3 square which we'll use to determine
# the vector components when we create a quiver plot.
#
# 1 2 3     UL UM UR
# 4 5 6     ML CC MR
# 7 8 9     LL LM LR
directions = OrderedDict()
directions[(0, 0)] = 1  # upper-left
directions[(0, 1)] = 2  # upper-middle
directions[(0, 2)] = 3  # upper-right
directions[(1, 0)] = 4  # middle-left
directions[(1, 1)] = 5  # center square
directions[(1, 2)] = 6  # middle-right
directions[(2, 0)] = 7  # lower-left
directions[(2, 1)] = 8  # lower-middle
directions[(2, 2)] = 9  # lower-right


def check(**kwargs):
    """This function is used for checking the arguments of functions and if they're within the
    ranges and types. In general, most functions will be different, but since some of the
    functions use similar arguments, it makes sense for a common type and value check to be
    defined and accessible. Any expected argument not found to be with the right type and value
    raises a RuntimeError.

        :param kwargs: (dict) The dict of arguments to check.
        :return: None
    """
    for key, value in kwargs.iteritems():
        if key == 'stride':
            if type(value) is not int:
                raise RuntimeError('The `stride` must be an integer.')
            else:
                if value < 1:
                    raise RuntimeError('The `stride` must be a positive non-zero number.')
        elif key == 'acceptance_threshold':
            if not np.isreal(value):
                raise RuntimeError('The `acceptance_threshold` must be a float or an integer.')
            else:
                if value > 1:
                    raise RuntimeError('The `acceptance_threshold` must be <= 1.0.')
        elif key == 'sentinel_value':
            if not np.isreal(value):
                raise RuntimeError('The `sentinel_value` must be a float or an integer.')


def determine_direction(direction):
    """This function uses the directions defined to calculate the vector directions which
    will be used to create a quiver plot. By itself, this is relatively straightforward.
    However, due to how imshow plots data (i.e. 0 is in the top-left corner), and since
    this is the same format used to generate all subsequent Ramachandran plots, it's
    imperative the quiver plot matches that order, hence these vector magnitudes.

    The idea here is that the directions here are all relative to the center (5), where they
    point inward towards it. As such, after modification to match what imshow produces
    the actual directions are:

        7 4 1
        8 5 2
        9 6 3

    The locations of the 5s will eventually be superimposed with an x to mark the peak locations.

        :param direction: (int) An integer within the values 1 - 9 that will be used to determine
                          the corresponding directional vectors to generate a quiver plot and
                          determine the gradients of steepest descent.

        :return u, v: (int, int) A tuple of the directional vectors corresponding to the input
                      direction.
    """
    u = 0
    v = 0
    if direction == 1:
        u = -1
        v = -1
    elif direction == 2:
        u = -1
    elif direction == 3:
        u = -1
        v = 1
    elif direction == 4:
        v = -1
    elif direction == 6:
        v = 1
    elif direction == 7:
        u = 1
        v = -1
    elif direction == 8:
        u = 1
    elif direction == 9:
        u = 1
        v = 1
    return u, v


def tiled_grid(array, size=3):
    """This function creates a tiled array - i.e. an input 2D array surrounded in the center by 8 other
    copies of itself. The intended use of this is to deal with the periodic boundary conditions of some
    grids as this method would disavow having to calculate the indices for a given array.

        :param array: (2D array) The input 2D numpy array.
        :param size: (int) The number of tiles to create of the input array. The resulting output will
                     be 2D, and have the dimension `(array.shape[0]*3, array.shape[1]*3)`. Default value = 3.
        :return: (2D array)"""
    tiled_row = np.tile(array, size)
    return np.concatenate((tiled_row, tiled_row, tiled_row), axis=0)


def coarse_grain_hist(hist, stride=4, acceptance_threshold=0.25, sentinel_value=1):
    """Coarse-grain the histogram by creating a sub-histogram where each cell is the average of
    subarrays of size `stride**2`. The intended purpose of this function is to clean up any spurious and
    sparse cells in the input histogram by only accepting subarrays which are at least filled by a certain
    percentage (i.e. `acceptance_threshold`).

    This resulting "mask" is used as a filter on the original histogram to clean up its contents for further
    analysis / display (e.g. in a quiver plot, regular plot, etc.).

        :param hist: (2D array) The input 2D histogram whose coarse-grained mask will be determined.
        :param stride: (int) The amount of cells along each axis that comprise the sub-array. Default value = 4.
        :param acceptance_threshold: (real) The minimum percentage of a subarray of size `stride` x `stride`
                                     that must be filled. Otherwise, these subarrays are ignored. If this
                                     value is too low, more spurious and minor basins appear. Default value = 0.25.
        :param sentinel_value: (int) The value to use when removing spurious values. Useful if the input
                               histogram is not logarithmic. Default value = 1.
        :return (cg_hist, masked_hist): (2D array, 2D array) The coarse-grained and "masked" histograms.
    """
    check(stride=stride, mean_threshold=acceptance_threshold, sentinel_value=sentinel_value)

    indices = np.arange(0, hist.shape[0]).tolist()
    boundaries = indices[::stride]
    threshold = (1 - acceptance_threshold)*stride**2

    cg_dim = hist.shape[0]/stride
    cg_hist = np.zeros(shape=(cg_dim, cg_dim))
    masked_hist = np.zeros_like(shape=hist.shape)

    for i_index, i in enumerate(boundaries):
        for j_index, j in enumerate(boundaries):
            # update our indices
            new_i = i + stride
            new_j = j + stride

            # Create a subset of the histogram that conforms to the chosen
            # stride.
            chunk = hist[i:new_i, j:new_j]

            # First, we remove all superfluous values - i.e. set them to some
            # sentinel value, and then set those to zero.
            chunk[np.where(chunk == np.nan)] = sentinel_value
            chunk[np.where(chunk == -np.nan)] = sentinel_value
            chunk[np.where(chunk == np.inf)] = sentinel_value
            chunk[np.where(chunk == -np.inf)] = sentinel_value
            find_zeros = chunk[np.where(chunk == sentinel_value)]
            chunk[np.where(chunk == sentinel_value)] = 0

            # Finally, we exclude any chunks which do not meet our
            # desired occupancy threshold. For e.g. if using a stride of 4,
            # the chunk area will be 16. If we want to exclude chunks that are
            # less than 25% occupied (which will clean up the grid), the number
            # of found zeroes will be 12. Any chunk that meets this criterion will
            # be excluded.
            if len(find_zeros) <= threshold:
                cg_hist[i_index, j_index] = chunk.mean()
                masked_hist[i:new_i, j:new_j] = hist[i:new_i, j:new_j]
    return cg_hist, masked_hist


def determine_histogram_directions(histogram):
    """Given an input histogram, determine the gradients of steepest descent and their associated vectors.

    For histograms with periodic-boundaries, this is done by creating a tiled array (i.e. the original array
    is in the center of a 3 x 3 grid filled with 9 copies of itself), and then using extracting a subarray that
    contains 1 row and column of the edges original array. For e.g. if our input array is:

        1 2 3
        4 5 6
        7 8 9

    The tiled array will be:

         1 2 3 1 2 3 1 2 3
         4 5 6 4 5 6 4 5 6
         7 8 9 7 8 9 7 8 9
         1 2 3 1 2 3 1 2 3
         4 5 6 4 5 6 4 5 6
         7 8 9 7 8 9 7 8 9
         1 2 3 1 2 3 1 2 3
         4 5 6 4 5 6 4 5 6
         7 8 9 7 8 9 7 8 9

    And, our "extended" array will be:

         9 7 8 9 7
         3 1 2 3 1
         6 4 5 6 4
         9 7 8 9 7
         3 1 2 3 1

    This ensures that when each 3 x 3 subarray is selected, the center element is always the nth element of
    the histogram, from which we compare the values of the other 8 directions to determine the gradients.

        :param histogram: (2D array) The input 2D histogram.
        :return array_directions, u, v: (2D array, 2D array, 2D array) The array directions, and the u and v
                                        vector 2D grids.
    """
    tiled_hist = tiled_grid(histogram)
    start = histogram.shape[0] - 1
    end = histogram.shape[0] * 2 + 1
    extended_hist = tiled_hist[start:end, start:end]
    indices = np.arange(0, extended_hist.shape[0])

    array_directions = np.zeros(histogram.shape).astype('int')
    u = np.zeros(histogram.shape).astype('int')
    v = np.zeros(histogram.shape).astype('int')
    for i in indices[:-2]:
        for j in indices[:-2]:
            # We use a size of 3 since we're only looking at a 3 x 3
            # subarray of the extended array. This is because we want
            # to determine the gradient of steepest descent local to
            # that point.
            new_i = i + 3
            new_j = j + 3

            # Create our chunk, and then determine where the location
            # of the lowest value is located.
            chunk = extended_hist[i:new_i, j:new_j]
            locs = np.where(chunk == np.nanmin(chunk))

            # We only check chunks whose center index is non-zero.
            if chunk[1, 1] != 0.0:
                for x, y in zip(locs[0], locs[1]):
                    # determine the directions based on the coordinate of the smallest
                    # element relative to the center (i.e. the current element).
                    coords = (x, y)
                    array_directions[i, j] = directions[coords]
                    u[i, j], v[i, j] = determine_direction(directions[coords])
    return array_directions, u, v


def boundaries_mask(histogram, basin_boundaries, bin_size):
    """Given an input 2D histogram and its basin boundaries obtained by coarse-graining the histogram,
    extract the corresponding indices to the original histogram.

        :param histogram: (2D array) The original 2D histogram.
        :param basin_boundaries: (OrderedDict) The basin boundaries (key: index, value: a list of 2D cartesian points)
                                  obtained from a coarse-grained representation of the original histogram.
        :param bin_size: (real) The bin_size used to generate the original histogram.
        :return scaled_basins, basins_scaled: (OrderedDict, OrderedDict) A dictionary where the keys are basin numbers
                                              and the values are the coordinates. And, a dictionary where the
                                              coordinates are the keys, and the values are the basin numbers.
    """
    indices = np.arange(0, histogram.shape[0])  # `histogram.shape` should be 144
    stride = int(10 / bin_size)
    boundaries = indices[::stride]  # every `stride` items
    scaled_boundaries = boundaries / stride

    # since the boundaries are just scaled by the stride, we can go through the basin boundaries
    scaled_basins = OrderedDict()  # this is for the basins by the points
    basins_scaled = OrderedDict()  # this is for the points by the basins

    # the idea here is to use the scaled basins to quickly determine
    for basin in basin_boundaries:
        basin_boundary = basin_boundaries[basin]
        scaled_basins[basin] = list()

        for x, y in basin_boundary:
            i = scaled_boundaries.index(x)
            j = scaled_boundaries.index(y)

            # Obtain the indices for the original array that were "hidden" by the `stride`.
            # This gets the x and y coordinates
            sbi = np.array(([boundaries[i]] * stride) + np.arange(0, stride, 1)).tolist()
            sbj = np.array(([boundaries[j]] * stride) + np.arange(0, stride, 1)).tolist()

            # Now save the adjusted coordinates.
            for si, sj in product(sbi, sbj):
                coordinate = (si, sj)
                scaled_basins[basin].append(coordinate)
                basins_scaled[coordinate] = basin
    return scaled_basins, basins_scaled


def partition(cg_hist):
    """This function segments a coarse-grained histogram by gradients of steepest descent. Basin centers, or regions of
    local minima, define basin centers, whose envelopes, relative areas and weights are determined.

        :param cg_hist: (2D array) The coarse-grained array.
        :return: (OrderedDict) An OrderedDict whose key is the basin number and value is the basin information stored
                  as a namedtuple.
    """
    x_dim, y_dim = cg_hist.shape
    array_directions, u, v = determine_histogram_directions(cg_hist)

    centers = OrderedDict()
    basins = OrderedDict()
    areas = OrderedDict()
    for x in range(0, array_directions.shape[0]):
        for y in range(0, array_directions.shape[0]):
            if array_directions[x, y] > 0:
                xx = copy(x)
                yy = copy(y)
                line = list()
                found_center = False

                # This loop will search for a basin center by following
                # the available lines
                line.append((xx, yy))  # start at the current point
                while not found_center:
                    # determine the indices for the next element as indicated
                    # by the directions of the current element in the quiver
                    # vectors
                    nx = xx + u[xx, yy]
                    ny = yy + v[xx, yy]
                    if nx == -1:
                        nx = x_dim - 1
                    if nx == x_dim:
                        nx = 0
                    if ny == -1:
                        ny = y_dim
                    if ny == y_dim:
                        ny = 0
                    xx = nx
                    yy = ny
                    line.append((xx, yy))

                    # check to see if the next element is a center
                    if array_directions[xx, yy] == 5:
                        found_center = True

                # The center is always the last point along the line.
                # Using this fact, and that the lines will contain duplicate indices
                # we simplify the found indices using a set.
                center = line[-1]
                if center not in centers:
                    centers[center] = list(set(line))
                else:
                    centers[center] = list(set(line + centers[center]))
                    areas[center] = len(centers[center])

    num_center = 1
    for center in centers:
        relative_area = float(len(areas[center]))/sum(areas.values())
        basins[num_center] = Basin(center=center,
                                   indices=centers[center],
                                   area=len(areas[center]),
                                   relative_area=relative_area,
                                   weight=None)
        num_center += 1
    return basins
