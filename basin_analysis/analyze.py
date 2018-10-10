import numpy as np
from copy import copy
from itertools import product
from collections import OrderedDict, namedtuple


# These are the functions are allowed to be exported. (The others are used internally, and offer no
# tangible benefit external to this solution.)
__all__ = 'tiled_grid,coarse_grain_hist,determine_histogram_directions,partition,determine_basin_attributes'.split(',')


# A lightweight read-only data structure to save and access basin information for a given
# simulation.
Basin = namedtuple('Basin', 'center rotated_center indices raw_indices relative_area relative_weight')


# Define the directions for a 3 x 3 square which we'll use to determine
# the vector components when we create a quiver plot.
#
# 1 2 3     UL UM UR
# 4 5 6     ML CC MR
# 7 8 9     LL LM LR
directions = OrderedDict()
directions[(0, 0)] = 1  # upper-left    (UL)
directions[(0, 1)] = 2  # upper-middle  (UM)
directions[(0, 2)] = 3  # upper-right   (UR)
directions[(1, 0)] = 4  # middle-left   (ML)
directions[(1, 1)] = 5  # center square (CC)
directions[(1, 2)] = 6  # middle-right  (MR)
directions[(2, 0)] = 7  # lower-left    (LL)
directions[(2, 1)] = 8  # lower-middle  (LM)
directions[(2, 2)] = 9  # lower-right   (LR)


def check(**kwargs):
    """This function is used for checking the arguments of functions and if they're within the
    ranges and types. In general, most functions will be different, but since some of the
    functions use similar arguments, it makes sense for a common type and value check to be
    defined and accessible. Any expected argument not found to be with the right type and value
    raises a RuntimeError.

        :param kwargs: (dict) The dict of arguments to check.
        :return: None
    """
    for key, value in kwargs.items():
        if key == 'cg_stride':
            if type(value) not in [int, tuple, list]:
                raise RuntimeError('The `cg_stride` must be an integer, or a tuple / list of size 2.')
            else:
                if type(value) in [tuple, list] and len(value) != 2:
                    raise RuntimeError('If `cg_stride` is a tuple / list, its length must be 2.')
                elif type(value) in [tuple, list] and type(value[0]) != int and type(value[1]) != int:
                    raise RuntimeError('The dimensions of the `cg_stride` must contain integers.')
                elif type(value) is int:
                    if value < 1:
                        raise RuntimeError('The `cg_stride` must be a positive non-zero number.')
        elif key == 'acceptance_threshold':
            if type(value) is not float:
                raise RuntimeError('The `acceptance_threshold` must be a float or an integer.')
            else:
                if value > 1:
                    raise RuntimeError('The `acceptance_threshold` must be <= 1.0.')
        elif key == 'sentinel_value':
            if type(value) not in [float, int]:
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


def coarse_grain_hist(hist, cg_stride, acceptance_threshold=0.25, sentinel_value=1):
    """Coarse-grain the histogram by creating a sub-histogram where each cell is the average of
    subarrays of size `cg_stride**2` (if `cg_stride` is an int), or `cg_stride[0] * cg_stride[1]` if the `cg_stride`
    is a tuple / list of size 2. The intended purpose of this function is to clean up any spurious and
    sparse cells in the input histogram by only accepting subarrays which are at least filled by a certain
    percentage (i.e. `acceptance_threshold`).

    This resulting "mask" is used as a filter on the original histogram to clean up its contents for further
    analysis / display (e.g. in a quiver plot, regular plot, etc.).

        :param hist: (2D array) The input 2D histogram whose coarse-grained mask will be determined.
        :param cg_stride: (int, tuple, list) This determines the size of the coarse-graining subarray. Valid options are
                                             an integer, or a tuple / list of size 2 that contains integers which
                                             corresponds to the x and y strides, respectively.
        :param acceptance_threshold: (real) The minimum percentage of a subarray of size `cg_stride` ** 2 (if
                                            `cg_stride` is an integer, or `cg_stride[0] * cg_stride[1]` if `cg_stride`
                                            is a tuple / list of size 2. Default value = 0.25.
                                     that must be filled. Otherwise, these subarrays are ignored. If this
                                     value is too low, more spurious and minor basins appear. Default value = 0.25.
        :param sentinel_value: (int) The value to use when removing spurious values. Useful if the input
                               histogram is not logarithmic. Default value = 1.
        :return (cg_hist, masked_hist): (2D array, 2D array) The coarse-grained and "masked" histograms.
    """
    check(cg_stride=cg_stride, acceptance_threshold=acceptance_threshold, sentinel_value=sentinel_value)
    x_stride = copy(cg_stride)
    y_stride = copy(cg_stride)
    if type(cg_stride) in [tuple, list]:
        x_stride, y_stride = cg_stride

    x_indices = np.arange(0, hist.shape[0], dtype=int).tolist()
    y_indices = np.arange(0, hist.shape[1], dtype=int).tolist()
    x_boundaries = x_indices[::x_stride]
    y_boundaries = y_indices[::y_stride]

    # The threshold as an integer - this will be used to determine the number of occupied spurious cells.
    threshold = int(np.floor((1 - acceptance_threshold) * x_stride * y_stride))

    # Create the coarse-grained and masked histograms
    cg_dimx = hist.shape[0] // x_stride
    cg_dimy = hist.shape[1] // y_stride
    cg_hist = np.zeros(shape=(cg_dimx, cg_dimy), dtype=float)
    masked_hist = np.zeros_like(hist)

    for i_index, i in enumerate(x_boundaries):
        for j_index, j in enumerate(y_boundaries):
            # update our indices
            new_i = i + x_stride
            new_j = j + y_stride

            # Create a subset of the histogram that conforms to the chosen
            # stride.
            chunk = hist[i:new_i, j:new_j]

            # First, we remove all spurious values - i.e. set them to some
            # sentinel value, and then set those to zero. This is necessary
            # if a logarithm has been applied, sometimes `np.nan` and `np.inf`
            # values will appear. These cells shouldn't be discounted, and
            # should be included nonetheless. However, since keeping them at
            # those values in the array would affect the determination of
            # gradients of steepest descent, we reassign them to 0.
            chunk[np.where(chunk == np.nan)] = sentinel_value
            chunk[np.where(chunk == -np.nan)] = sentinel_value
            chunk[np.where(chunk == np.inf)] = sentinel_value
            chunk[np.where(chunk == -np.inf)] = sentinel_value
            find_spurious = chunk[np.where(chunk == sentinel_value)]
            chunk[np.where(chunk == sentinel_value)] = 0

            # Finally, we exclude any chunks which do not meet our
            # desired occupancy threshold. For e.g. if using a coarse-grained stride of 4
            # for both dimensions, the chunk area will be 16. If we want to exclude chunks
            # that are less than 25% occupied (which will clean up the grid), the number
            # of found zeroes will be 12. Any chunk that meets this criterion will
            # be excluded.
            if len(find_spurious) <= threshold:
                cg_hist[i_index, j_index] = chunk.mean()
                masked_hist[i:new_i, j:new_j] = hist[i:new_i, j:new_j]
    return cg_hist, masked_hist


def determine_histogram_directions(histogram, periodic_boundary=True):
    """Given an input histogram, determine the gradients of steepest descent and their associated vectors. This
    is intended for use on a coarse-grained histogram, however any histogram can be used.

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
        :param periodic_boundary: (bool) Whether or not to consider periodic boundary conditions. Default = True.
        :return array_directions, u, v: (2D array, 2D array, 2D array) The array directions, and the u and v
                                        vector 2D grids.
    """
    tiled_hist = tiled_grid(histogram)
    x_start = histogram.shape[0] - 1
    x_end = histogram.shape[0] * 2 + 1
    y_start = histogram.shape[1] - 1
    y_end = histogram.shape[1] * 2 + 1
    extended_hist = tiled_hist[x_start:x_end, y_start:y_end]

    # The extended histogram was created so that for periodic boundaries every cell would be visited (due to the 3 x 3
    # sub-array). However, to utilize the same algorithm for non-periodic boundaries we simply have to set the extended
    # regions (i.e. the first and last columns and rows) to 0. Doing so would create an artificial cutoff, but will also
    # allow for the segments to be determined individually with the current algorithm.
    if not periodic_boundary:
        extended_hist[0, :] = 0  # first row
        extended_hist[-1, :] = 0  # last row
        extended_hist[:, 0] = 0  # first column
        extended_hist[:, -1] = 0  # last column

    x_indices = np.arange(0, extended_hist.shape[0], dtype=int)
    y_indices = np.arange(0, extended_hist.shape[1], dtype=int)

    array_directions = np.zeros(histogram.shape, dtype=int)
    u = np.zeros(histogram.shape, dtype=int)
    v = np.zeros(histogram.shape, dtype=int)
    for i in x_indices[:-2]:
        for j in y_indices[:-2]:
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
                    # Determine the directions based on the coordinate of the smallest
                    # element relative to the center (i.e. the current element).
                    coords = (x, y)
                    array_directions[i, j] = directions[coords]
                    u[i, j], v[i, j] = determine_direction(directions[coords])
    return array_directions, u, v


def boundaries_mask(histogram, cg_basin_boundaries, cg_stride):
    """Given an input 2D histogram and its basin boundaries obtained by coarse-graining the histogram,
    extract the corresponding indices to the original histogram.

        :param histogram: (2D array) The original 2D histogram.
        :param cg_basin_boundaries: (OrderedDict) The caorse-grained basin boundaries (key: index, value: a list of 2D
                                    cartesian points) obtained from a coarse-grained representation of the original
                                    histogram.
        :param cg_stride: (int, tuple, list) This determines the size of the coarse-graining subarray. Valid options are
                                             an integer, or a tuple / list of size 2 that contains integers which
                                             corresponds to the x and y strides, respectively.
        :return scaled_basins, basins_scaled: (OrderedDict, OrderedDict) A dictionary where the keys are basin numbers
                                              and the values are the coordinates. And, a dictionary where the
                                              coordinates are the keys, and the values are the basin numbers.
    """
    check(cg_stride=cg_stride)
    x_stride = copy(cg_stride)
    y_stride = copy(cg_stride)
    if type(cg_stride) in [tuple, list]:
        x_stride, y_stride = cg_stride

    # Convert the indices to a list for simplicity
    x_indices = np.arange(0, histogram.shape[0], dtype=int)
    y_indices = np.arange(0, histogram.shape[1], dtype=int)
    x_boundaries = x_indices[::x_stride]  # every `x_stride` items
    y_boundaries = y_indices[::y_stride]  # every `y_stride` items
    x_scaled_boundaries = (x_boundaries / x_stride).tolist()
    y_scaled_boundaries = (y_boundaries / y_stride).tolist()

    # Since the boundaries are just scaled by the stride, we can go through the basin boundaries
    # to populate the indices for the centers.
    scaled_basins = OrderedDict()  # this is for the basins by the points
    basins_scaled = OrderedDict()  # this is for the points by the basins

    # The idea here is to use the scaled basins to quickly determine which points in the original
    # histogram correspond to those in the coarse-grained histogram.
    for basin in cg_basin_boundaries:
        basin_boundary = cg_basin_boundaries[basin]
        scaled_basins[basin] = list()

        for x, y in basin_boundary:
            i = x_scaled_boundaries.index(x)
            j = y_scaled_boundaries.index(y)

            # Obtain the indices for the original array that were "hidden" by the `stride` and keep them.
            # This gets the "scaled" x and y coordinates. For example, if the stride is 4, the `x_indices` would be
            # `0, 4, 8, ...`. Therefore, the "hidden" indices would be those between the indices - e.g.
            # `1,2,3,5,6,7,...`. These indices are then added to the original "stride" indices to ensure that we
            # capture all the coordinates.
            sbi = np.array(([x_boundaries[i]] * x_stride) + np.arange(0, x_stride, 1), dtype=int).tolist()
            sbj = np.array(([y_boundaries[j]] * y_stride) + np.arange(0, y_stride, 1), dtype=int).tolist()

            # Now save the adjusted coordinates.
            for si, sj in product(sbi, sbj):
                coordinate = (si, sj)
                scaled_basins[basin].append(coordinate)
                basins_scaled[coordinate] = basin
    return scaled_basins, basins_scaled


def partition(cg_hist, periodic_boundary=True):
    """This function segments a coarse-grained histogram by gradients of steepest descent. Basin centers, or regions of
    local minima, define basin centers, whose envelopes, relative areas and weights are determined.

        :param cg_hist: (2D array) The coarse-grained array.
        :param periodic_boundary: (bool) Whether or not to consider periodic boundary conditions. Default = True.
        :return: (OrderedDict) An OrderedDict whose key is the basin number and value is the basin information stored
                  as a namedtuple.
    """
    x_dim, y_dim = cg_hist.shape
    array_directions, u, v = determine_histogram_directions(cg_hist, periodic_boundary)

    cg_basin_boundaries = OrderedDict()
    cg_basin_areas = OrderedDict()
    for x in range(0, array_directions.shape[0]):
        for y in range(0, array_directions.shape[1]):
            if array_directions[x, y] not in [-1, 0]:
                xx = copy(x)
                yy = copy(y)
                line = list()
                found_center = False

                # This loop will search for a basin center by following
                # the available lines.
                line.append((xx, yy))  # start at the current point
                while not found_center:
                    # Determine the indices for the next element as indicated
                    # by the directions of the current element in the quiver
                    # vectors.
                    nx = xx + u[xx, yy]
                    ny = yy + v[xx, yy]
                    if nx == -1:
                        nx = x_dim - 1
                    if nx == x_dim:
                        nx = 0
                    if ny == -1:
                        ny = y_dim - 1
                    if ny == y_dim:
                        ny = 0
                    xx = nx
                    yy = ny
                    line.append((xx, yy))

                    # Check to see if the next element is a center.
                    if array_directions[xx, yy] == 5:
                        found_center = True

                # The center is always the last point along the line.
                # Using this fact, and that the lines will contain duplicate indices
                # we simplify the found indices using a set.
                center = line[-1]

                if center not in cg_basin_boundaries:
                    cg_basin_boundaries[center] = list(set(line))
                else:
                    cg_basin_boundaries[center] += line
                    cg_basin_boundaries[center] = list(set(cg_basin_boundaries[center]))
                cg_basin_areas[center] = len(cg_basin_boundaries[center])
    return cg_basin_boundaries, cg_basin_areas


def determine_basin_attributes(raw_hist, pmf_hist, cg_hist, periodic_boundary=True):
    """This function determines the basin attributes: centers, indices, relative areas, and relative weights
    from the raw histogram (i.e. no PMF applied), the PMF applied histogram, and the coarse-grained histogram.

    This is done by coarse-graining the histogram to determine the relative basins, and their areas. Since this
    also screens away the minor basins, the total weight will always be less than or equal to the `raw_hist`. Using
    this mask we can then determine the basin attributes.

        :param raw_hist: (2D array) The original array with no PMF applied - i.e. just raw counts.
        :param pmf_hist: (2D array) The PMF applied histogram.
        :param cg_hist: (2D array) The coarse-grained histogram. This has dimensions of `raw_hist.shape` / `scale`.
        :param periodic_boundary: (bool) Whether or not to consider periodic boundary conditions. Default = True.
        :return basins: (OrderedDict) An OrderedDict containing the basins identified by number, associated with
                        its corresponding attributes: centers, indices, relative areas and relative weights.
    """
    # For the `raw_hist` the total weight is actually the number of steps - e.g. 3.5M x 50 sims = 175K (at every 10K).
    # However, since we're using the mask to populate the basins, this number will be less than or equal to
    # that value (175K in our example) since the histogram will be screened / masked.
    x_stride = pmf_hist.shape[0] // cg_hist.shape[0]
    y_stride = pmf_hist.shape[1] // cg_hist.shape[1]
    cg_stride = (x_stride, y_stride)
    total_area = 0
    total_weight = 0
    basin_counts = OrderedDict()

    # Find the basin boundaries through partitioning.
    cg_basin_boundaries, cg_basin_areas = partition(cg_hist, periodic_boundary)
    scaled_basins, basins_scaled = boundaries_mask(raw_hist, cg_basin_boundaries, cg_stride)

    # Find the values of the pixels within the basins.
    for basin in scaled_basins:
        basin_counts[basin] = list()

    # Determine the occupied cells of the raw histogram.
    occupied_x, occupied_y = np.where(raw_hist != 0.0)
    
    # Now, determine which basins these belong to and save their counts / values.
    for x, y in zip(occupied_x, occupied_y):
        if (x, y) in basins_scaled:
            basin = basins_scaled[(x, y)]
            basin_counts[basin].append(raw_hist[x, y])
            total_weight += raw_hist[x, y]
            total_area += 1

    # Now, populate the corresponding basin attributes: center, indices, relative area, relative weight.
    basins = OrderedDict()
    total_area = sum(cg_basin_areas.values())
    running_total = 0
    num_center = 1
    for basin_center in basin_counts:
        area_indices = np.array(scaled_basins[basin_center])
        relative_area = len(area_indices) / float(total_area)
        relative_weight = raw_hist[area_indices[:, 0], area_indices[:, 1]].sum() / float(total_weight)

        # Build the `Basin` object using the information we've determined.
        # Of particular note: `rotated_center` is the index of the center if the histogram has been
        # rotated 90 degrees (i.e. `np.rot90`). The -1 is due to the shape not being zero-indexed.
        basins[num_center] = Basin(center=basin_center,
                                   rotated_center=(basin_center[0], cg_hist.shape[1] - basin_center[1] - 1),
                                   indices=cg_basin_boundaries[basin_center],
                                   raw_indices=area_indices,
                                   relative_area=relative_area,
                                   relative_weight=relative_weight)
        running_total += relative_weight
        num_center += 1
    return basins
