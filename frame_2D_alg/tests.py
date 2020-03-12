"""
Script for testing 2D alg.
Change quickly in parallel with development.
"""

from time import time
import operator as op
from comp_pixel import (
    comp_pixel,
    comp_pixel_old,
    comp_pixel_ternary,
    comp_3x3, comp_3x3_loop,
)
from intra_comp import (
    angle_diff, translated_operation,
    calc_a, comp_a,
)
from test_sets import (
    comp_pixel_test_pairs,
    calc_a_test_pairs,
    pixels, rderts, gderts, angles,
)
from utils import *


# ----------------------------------------------------------------------------
# Constants

# ----------------------------------------------------------------------------
# Functions

def test_function(func, test_pairs, eval_func=op.eq, **kwargs):
    """
    Meta test function. Run output tests for a function.
    Stop if there's an unexpected output.
    Parameters
    ----------
    func : callable
        The function to test.
    test_pairs : list of tuples
        List of inp, out pairs, where out is the expected return
        value from call of the tested function with inp as input.
        Type of inp:
            list, tuple or set : inp is passed as arguments.
            dict : inp is passed as keyword arguments.
            other types: inp is passed as a single input.
    eval_func : callable, optional
        Function used to check return value against the expected
        output. eval_func must take at least two required arguments
        and return a bool.
    **kwargs
        Keyword arguments to be passed to eval_func
    """
    start_time = time()
    print(f'Testing {func.__str__()}... ', end='')
    for inp, out in test_pairs:
        if isinstance(inp, (list, tuple, set)):
            o = func(*inp)
        elif isinstance(inp, dict):
            o = func(**inp)
        else:
            o = func(inp)

        if not eval_func(o, out, **kwargs):
            print('\n\nTest failed on:')
            print('Input:')
            print(inp)
            print('Expected output:')
            print(out)
            print('Actual output:')
            print(o)
            break
    else:
        print('Test success!')

    time_elapsed = time() - start_time
    test_cnt = len([*test_pairs])
    print(f'Finished {test_cnt} '
          f'{"tests" if test_cnt > 1 else "test"} '
          f'in {time_elapsed} seconds.\n')


# ----------------------------------------------------------------------------
# Module utils' test functions

def test_is_close(
        test_pairs=(
                (('asd', ''), False),
                ((1235, 1234.99), True),
                ((7, 'a'), False),
                ((np.array([12.1999]), np.array([12.2])), True),
                (('Dong Thap', 'Dong thap'), False),
                (('can tho', 'can tho'), True),
                ((np.nan, np.nan), False),
                ({'x1':np.nan, 'x2':np.nan, 'equal_nan':True}, True),
                ((np.array([1.00001, np.nan]),
                  np.array([1, np.nan])),
                 False),
                ({'x1':np.array([1.00001, np.nan]),
                  'x2':np.array([1, np.nan]),
                  'equal_nan':True},
                 True),
        ),
):
    """Run tests for is_close function in utils."""
    test_function(is_close, test_pairs)


# ----------------------------------------------------------------------------
# Module comp_pixel test functions

def test_comp_pixel(
        func=comp_pixel,
        test_pairs=comp_pixel_test_pairs,
        eval_func=is_close,  # To check float equality
):
    """Test comp_pixels."""
    test_function(func, test_pairs, eval_func)


def test_comp_3x3(
        test_pairs=zip(pixels, rderts),
        eval_func=is_close,
        loop=False,
):
    test_function(comp_3x3_loop if loop else comp_3x3,
                  test_pairs, eval_func)

# ----------------------------------------------------------------------------
# Module frame_blobs test functions


# ----------------------------------------------------------------------------
# Module intra_comp test functions

def test_translated_operation(
        test_pairs=(
                ( # rng = 0, with subtraction as the operator
                        # inputs
                        (np.array([[0, 1, 2],
                                   [3, 4, 5],
                                   [6, 7, 8]]),
                         0, op.sub, # rng = 1 and operator = subtract
                         ),
                        # output
                        np.array([[[-2, 4], [-2, 4]],
                                  [[-2, 4], [-2, 4]]]),
                ),
                ( # rng = 1, with subtraction as the operator
                        # inputs
                        (np.array([[0, 1, 2],
                                   [3, 4, 5],
                                   [6, 7, 8]]),
                         1, op.sub, # rng = 1 and operator = subtract
                         ),
                        # output
                        np.array([[[8, 6, 4, -2]]]),
                        # output from older version
                        # np.array([[[-4, -3, -2, 1, 4, 3, 2, -1]]]),
                ),
                ( # rng = 0, with addition as the operator
                        # inputs
                        (np.array([[0, 1, 2],
                                   [3, 4, 5],
                                   [6, 7, 8]]),
                         0, op.add, # rng = 1 and operator = subtract
                         ),
                        # output
                        np.array([[[4, 4], [6, 6]],
                                  [[10, 10], [12, 12]]]),
                ),
                ( # rng = 1, with addition as the operator
                        # inputs
                        (np.array([[0, 1, 2],
                                   [3, 4, 5],
                                   [6, 7, 8]]),
                         1, op.add, # rng = 1 and operator = subtract
                         ),
                        # output
                        np.array([[[8, 8, 8, 8]]]),
                        # output from older version
                        # np.array([[[4, 5, 6, 9, 12, 11, 10, 7]]]),
                ),
                (  # rng = 1, with subtraction as the operator
                        # inputs
                        (np.array([[ 0,  1,  2,  3,  4,  5],
                                   [ 6,  7,  8,  9, 10, 11],
                                   [12, 13, 14, 15, 16, 17],
                                   [18, 19, 20, 21, 22, 23],
                                   [24, 25, 26, 27, 28, 29]]),
                         1, op.sub,  # rng = 1 and operator = subtract
                         ),
                        # output
                        np.array([[[14, 12, 10, -2], [14, 12, 10, -2],
                                   [14, 12, 10, -2], [14, 12, 10, -2]],
                                  [[14, 12, 10, -2], [14, 12, 10, -2],
                                   [14, 12, 10, -2], [14, 12, 10, -2]],
                                  [[14, 12, 10, -2], [14, 12, 10, -2],
                                   [14, 12, 10, -2], [14, 12, 10, -2]]]),
                ),
        ),
        eval_func=is_close,
):
    """Test translated_operation."""
    test_function(translated_operation, test_pairs, eval_func)


def test_angle_diff(
        test_pairs=(
                ( # 0 - 90 = -90 degrees
                        # inputs
                        ((0, 1),  # 0 degrees
                         (1, 0)), # 90 degrees
                        # output
                        (-1, 0), # -90 degrees
                ),
                ( # 135 - 90 = 45 degrees
                        # inputs
                        ((0.5**0.5, -0.5**0.5), # 135 degrees
                         (1, 0)), # 90 degrees
                        # output
                        (0.5**0.5, 0.5**0.5), # 45 degrees
                ),
                ( # 90 - 60 = 30 degrees
                        # inputs
                        ((1, 0), # 90 degrees
                         (0.75**0.5, 0.5)), # 60 degrees
                        # output
                        (0.5, 0.75**0.5), # 30 degrees
                ),
                ( # arrays of angles
                        # inputs
                        (
                                # a2
                                np.array((
                                        (0, 1), # 0 degrees
                                        (0.5**0.5, -0.5**0.5), # 135 degrees
                                        (1, 0), # 90 degrees
                                )).T,

                                # a1
                                np.array((
                                        (1, 0), # 90 degrees
                                        (1, 0), # 90 degrees
                                        (0.75**0.5, 0.5), # 60 degrees
                                )).T,
                        ),

                        # output
                        np.array((
                                (-1, 0), # -90 degrees
                                (0.5**0.5, 0.5**0.5), # 45 degrees
                                (0.5, 0.75**0.5), # 60 degrees
                        )).T,
                ),
        ),
        eval_func=is_close,
):
    """Test angle_diff."""
    test_function(angle_diff, test_pairs, eval_func)

def test_calc_a(
        test_pairs=calc_a_test_pairs,
        eval_func=is_close,
):
    test_function(calc_a, test_pairs, eval_func, equal_nan=True)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
    # Put tests here
    print(comp_a(gderts[2], rng=0, inp=(1, 2, 3)))