"""This file contains functions and code that provide useful utilities when building the related algorithm and code."""

__all__ = 'check'


def check_cg_stride(cg_stride):
    """Check whether the `cg_stride` is the right type and values.
        :param cg_stride: (int, tuple, list) The dimensions / value by which to create
                                             subarrays that will coarse-grain some input histogram.
        :return: None
    """
    if type(cg_stride) not in [int, tuple, list]:
        raise RuntimeError('The `cg_stride` must be an integer, or a tuple / list of size 2.')
    else:
        if type(cg_stride) in [tuple, list] and len(cg_stride) != 2:
            raise RuntimeError('If `cg_stride` is a tuple / list, its length must be 2.')
        elif type(cg_stride) in [tuple, list] and type(cg_stride[0]) != int and type(cg_stride[1]) != int:
            raise RuntimeError('The dimensions of the `cg_stride` must contain integers.')
        elif type(cg_stride) is int:
            if cg_stride < 1:
                raise RuntimeError('The `cg_stride` must be a positive non-zero number.')


def check_acceptance_threshold(ratio):
    """Check whether the `acceptance_threshold` / ratio is the right type and value.
        :param ratio: (float) The number of spurious cells within a chunk to accept, as a
                              percent.
        :return: None
    """
    if type(ratio) is not float:
        raise RuntimeError('The `acceptance_threshold` must be a float or an integer.')
    else:
        if ratio > 1:
            raise RuntimeError('The `acceptance_threshold` must be <= 1.0.')


def check_sentinel_value(sentinel_value):
    """Check whether the `sentinel_value` is the right type and value.
        :param sentinel_value: (real) The value which to use when removing spurious values.
        :return: None
    """
    if type(sentinel_value) not in [float, int]:
        raise RuntimeError('The `sentinel_value` must be a float or an integer.')


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
            check_cg_stride(value)

        elif key == 'acceptance_threshold':
            check_acceptance_threshold(value)

        elif key == 'sentinel_value':
            check_sentinel_value(value)
