# Useful references on using unittest:
# 1. https://stackoverflow.com/questions/6103825/how-to-properly-use-unit-testings-assertraises-with-nonetype-objects
import unittest
import numpy as np
import matplotlib.pyplot as plt
from basin_analysis.analyze import check, determine_direction, tiled_grid
from basin_analysis.analyze import coarse_grain_hist, determine_histogram_directions, partition


class TestProject(unittest.TestCase):
    def test_check_args_success(self):
        """Check whether the `check` function successfully completes."""
        stride = 4
        acceptance_threshold = 0.25
        sentinel_value = 1
        result = check(stride=stride,
                       acceptance_threshold=acceptance_threshold,
                       sentinel_value=sentinel_value)
        self.assertEqual(result, None)

    def test_check_args_type_failure_stride(self):
        """Check whether the `check` function successfully captures that `stride`
        is the wrong type (float instead of int)."""
        stride = 4.0
        acceptance_threshold = 0.25
        sentinel_value = 1
        with self.assertRaises(RuntimeError):
            check(stride=stride,
                  acceptance_threshold=acceptance_threshold,
                  sentinel_value=sentinel_value)

    def test_check_args_type_failure_acceptance_threshold(self):
        """Check whether the `check` function successfully captures that `acceptance_threshold`
        is the wrong type (list instead of float)."""
        stride = 4
        acceptance_threshold = [0.25]
        sentinel_value = 1
        with self.assertRaises(RuntimeError):
            check(stride=stride,
                  acceptance_threshold=acceptance_threshold,
                  sentinel_value=sentinel_value)

    def test_check_args_type_failure_sentinel_value(self):
        """Check whether the `check` function successfully captures that `sentinel_value`
        is the wrong type (list instead of int / float)."""
        stride = 4
        acceptance_threshold = 0.25
        sentinel_value = [1.0]
        with self.assertRaises(RuntimeError):
            check(stride=stride,
                  acceptance_threshold=acceptance_threshold,
                  sentinel_value=sentinel_value)

    def test_determine_direction(self):
        """Check that `determine_direction` actually returns the right vectors from
        `determine_direction`."""
        directions = range(1, 10)
        expected = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 0), (0, 1),
                    (1, -1), (1, 0), (1, 1)]
        for index, direction in enumerate(directions):
            self.assertEqual(determine_direction(direction), expected[index])

    def test_tiled_grid_shape(self):
        """Check that the shape of the array obtained via `tiled_grid` has the right dimensions."""
        xdim = 3
        ydim = 3
        array = np.arange(1, 10).reshape((xdim, ydim))
        tiled_array = tiled_grid(array, size=3)
        self.assertEqual(tiled_array.shape, (array.shape[0] * xdim, array.shape[1] * ydim))

    def test_tiled_grid_values(self):
        """Check that the array obtained via `tiled_grid` has been created correctly. This is done
        by checking the occurrences of the values of the input array."""
        xdim = 3
        ydim = 3
        array = np.arange(1, 10).reshape((xdim, ydim))
        tiled_array = tiled_grid(array, size=3)
        for value in np.arange(1, 10):
            counts = tiled_array[np.where(tiled_array == value)]
            self.assertEqual(len(counts), xdim * ydim)

    def test_coarse_grain_hist(self):
        """This tests whether we can coarse-grain an input histogram. To do so without data, we create a fake
        histogram and apply `coarse_grain_hist` on it. The thing to note here is that the `xdim` and `ydim`
        must be exactly divisible by the stride - i.e. yield an integer in each case.
        
        As for a more robust test - i.e. checking the resulting values and whether or not it's correct, will
        require some careful thought on the kind of data that's needed."""
        np.random.seed(0)  # for predictable results

        low = 0
        high = 3
        xdim = 9
        ydim = 9
        stride = 3
        expected_shape = (xdim//stride, ydim//stride)
        num_points = xdim * ydim
        hist = np.random.randint(low, high, num_points).reshape((xdim, ydim))  # this is our analogous "histogram"

        # Apply a PMF on the "histogram" to make it easier to work with since `coarse_grain_hist` expects to find
        # areas of steepest descent - if all the numbers are positive, this implementation of the algorithm won't
        # work.
        hist = -np.log(hist)
        hist[hist == np.inf] = 0
        hist[hist == -0.0] = 0
        cg_hist, masked_hist = coarse_grain_hist(hist, stride=stride, acceptance_threshold=0.25)

        self.assertEqual(cg_hist.shape, expected_shape)
        self.assertEqual(masked_hist.shape, hist.shape)

    def test_determine_histogram_directions_random_array(self):
        """Test the function `determine_histogram_directions`, for which, we need to generate a coarse-grained
        histogram. Similar to `test_coarse_grain_hist`, the dimensions of `xdim` and `ydim` must be exactly divisible
        by the stride - i.e. yield an integer - in each case.

        A more robust test with expected inputs and outputs will have to be devised at some point."""
        np.random.seed(0)  # for predictable results

        low = 0
        high = 3
        xdim = 9
        ydim = 9
        stride = 3
        num_points = xdim * ydim
        hist = np.random.randint(low, high, num_points).reshape((xdim, ydim))  # this is our analogous "histogram"

        # Apply a PMF on the "histogram" to make it easier to work with since `coarse_grain_hist` expects to find
        # areas of steepest descent - if all the numbers are positive, this implementation of the algorithm won't
        # work.
        hist = -np.log(hist)
        hist[hist == np.inf] = 0
        hist[hist == -0.0] = 0
        cg_hist, masked_hist = coarse_grain_hist(hist, stride=stride, acceptance_threshold=0.25)
        array_directions, u, v = determine_histogram_directions(cg_hist)

        # check the dimensions
        self.assertEqual(array_directions.shape, cg_hist.shape)
        self.assertEqual(u.shape, cg_hist.shape)
        self.assertEqual(v.shape, cg_hist.shape)

        # now, check that there are the right number of centers
        num_centers = len(array_directions[array_directions == 5])
        self.assertGreater(num_centers, 0)

    def test_boundaries_mask(self):
        """This function tests `boundaries_mask` - i.e. given some input histogram, we're able to obtain
        the corresponding array directions, and basin segments. To obtain predictable results, we'll use
        a 2D gaussian and several iterations to test the robustness of this technique.

        Ref: https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-79.php
        """
        stride = 4
        extent = 12
        x, y = np.meshgrid(np.linspace(-1, 1, extent), np.linspace(-1, 1, extent))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 1.0, 1.0

        # Using the above parameters, we create a gaussian histogram that's centered at `extent/2.0`.
        # Since the algorithm expects to find gradients of steepest descent, we need to invert the values (hence
        # the leading minus sign).
        gaussian_hist = -np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

        # Now we coarse-grain the histogram
        cg_hist, masked_hist = coarse_grain_hist(gaussian_hist, stride=stride, acceptance_threshold=0.25)
        array_directions, u, v = determine_histogram_directions(cg_hist)
        plt.imshow(array_directions, cmap='jet')
        plt.show()
        print(111)
        pass

    def test_partition(self):
        pass


if __name__ == '__main__':
    unittest.main()
