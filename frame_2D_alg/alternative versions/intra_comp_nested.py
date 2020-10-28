"""
Cross-comparison of pixels or gradients, in 2x2 or 3x3 kernels
"""

import numpy as np
import functools
from utils_nested import *

# Sobel coefficients to decompose ds into dy and dx:

YCOEFs = np.array([-1, -2, -1, 0, 1, 2, 1, 0])
XCOEFs = np.array([-1, 0, 1, 2, 1, 0, -1, -2])
''' 
    |--(clockwise)--+  |--(clockwise)--+
    YCOEF: -1  -2  -1  ¦   XCOEF: -1   0   1  ¦
            0       0  ¦          -2       2  ¦
            1   2   1  ¦          -1   0   1  ¦
            
Scharr coefs:
YCOEFs = np.array([-47, -162, -47, 0, 47, 162, 47, 0])
XCOEFs = np.array([-47, 0, 47, 162, 47, 0, -47, -162])
'''

def comp_r(dert__, ave, root_fia, mask=None):
    '''
    Cross-comparison of input param (dert[0]) over rng passed from intra_blob.
    This fork is selective for blobs with below-average gradient,
    where input intensity didn't vary much in shorter-range cross-comparison.
    Such input is predictable enough for selective sampling: skipping current
    rim derts as kernel-central derts in following comparison kernels.
    Skipping forms increasingly sparse output dert__ for greater-range cross-comp, hence
    rng (distance between centers of compared derts) increases as 2^n, with n starting at 0:
    rng = 1: 3x3 kernel,
    rng = 2: 5x5 kernel,
    rng = 4: 9x9 kernel,
    ...
    Due to skipping, configuration of input derts in next-rng kernel will always be 3x3, see:
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_diagrams.png
    '''
    i__ = dert__[0]  # i is pixel intensity

    '''
    sparse aligned i__center and i__rim arrays:
    '''
    i__center = i__[1:-1:2, 1:-1:2]  # also assignment to new_dert__[0]
    i__topleft = i__[:-2:2, :-2:2]
    i__top = i__[:-2:2, 1:-1:2]
    i__topright = i__[:-2:2, 2::2]
    i__right = i__[1:-1:2, 2::2]
    i__bottomright = i__[2::2, 2::2]
    i__bottom = i__[2::2, 1:-1:2]
    i__bottomleft = i__[2::2, :-2:2]
    i__left = i__[1:-1:2, :-2:2]
    ''' 
    unmask all derts in kernels with only one masked dert (can be set to any number of masked derts), 
    to avoid extreme blob shrinking and loss of info in other derts of partially masked kernels
    unmasked derts were computed due to extend_dert() in intra_blob   
    '''
    if mask is not None:
        majority_mask = ( mask[1:-1:2, 1:-1:2].astype(int)
                        + mask[:-2:2, :-2:2].astype(int)
                        + mask[:-2:2, 1:-1: 2].astype(int)
                        + mask[:-2:2, 2::2].astype(int)
                        + mask[1:-1:2, 2::2].astype(int)
                        + mask[2::2, 2::2].astype(int)
                        + mask[2::2, 1:-1:2].astype(int)
                        + mask[2::2, :-2:2].astype(int)
                        + mask[1:-1:2, :-2:2].astype(int)
                        ) > 1
    else:
        majority_mask = None  # returned at the end of function

    if root_fia:  # initialize derivatives:
        dy__ = np.zeros_like(i__center)  # sparse to align with i__center
        dx__ = np.zeros_like(dy__)
        m__ = np.zeros_like(dy__)

    else:  # root fork is comp_r, accumulate derivatives:
        dy__ = dert__[1][1:-1:2, 1:-1:2].copy()  # sparse to align with i__center
        dx__ = dert__[2][1:-1:2, 1:-1:2].copy()
        m__ = dert__[4][1:-1:2, 1:-1:2].copy()

    # compare four diametrically opposed pairs of rim pixels:

    d_tl_br = i__topleft - i__bottomright
    d_t_b = i__top - i__bottom
    d_tr_bl = i__topright - i__bottomleft
    d_r_l = i__right - i__left

    dy__ += (d_tl_br * YCOEFs[0] +
             d_t_b * YCOEFs[1] +
             d_tr_bl * YCOEFs[2] +
             d_r_l * YCOEFs[3])

    dx__ += (d_tl_br * XCOEFs[0] +
             d_t_b * XCOEFs[1] +
             d_tr_bl * XCOEFs[2] +
             d_r_l * XCOEFs[3])

    g__ = np.hypot(dy__, dx__) - ave  # gradient, recomputed at each comp_r
    '''
    inverse match = SAD, direction-invariant and more precise measure of variation than g
    (all diagonal derivatives can be imported from prior 2x2 comp)
    '''
    m__ += ave - ( abs(i__center - i__topleft)
                 + abs(i__center - i__top)
                 + abs(i__center - i__topright)
                 + abs(i__center - i__right)
                 + abs(i__center - i__bottomright)
                 + abs(i__center - i__bottom)
                 + abs(i__center - i__bottomleft)
                 + abs(i__center - i__left)
                 )

    return (i__center, dy__, dx__, g__, m__), majority_mask


def comp_a(dert__, ave, mask=None):  # cross-comp of angle in 2x2 kernels

    if mask is not None:
        majority_mask = (mask[:-1, :-1].astype(int) +
                         mask[:-1, 1:].astype(int) +
                         mask[1:, 1:].astype(int) +
                         mask[1:, :-1].astype(int)
                         ) > 1
    else:
        majority_mask = None

    i__, dy__, dx__, g__, m__ = dert__[:5]  # day__,dax__,ga__,ma__ are recomputed
    g__[np.where(g__ == 0)] = 1  # to avoid / 0
    a__ = [dy__, dx__] / (g__ + ave)  # angle, restore g to abs, similar to calc_a

    # each shifted a in 2x2 kernel
    a__topleft = a__[:, :-1, :-1]
    a__topright = a__[:, :-1, 1:]
    a__botright = a__[:, 1:, 1:]
    a__botleft = a__[:, 1:, :-1]

    # diagonal angle differences:
    sin_da0__, cos_da0__ = angle_diff(a__topleft, a__botright)
    sin_da1__, cos_da1__ = angle_diff(a__topright, a__botleft)

    ma__ = np.hypot(sin_da0__ + 1, cos_da0__ + 1) \
         + np.hypot(sin_da1__ + 1, cos_da1__ + 1)
    # ma = inverse angle match = SAD: covert sin and cos da to 0->2 range

    day__ = [-sin_da0__ - sin_da1__, cos_da0__ + cos_da1__]
    # angle change in y, sines are sign-reversed because da0 and da1 are top-down, no reversal in cosines

    dax__ = [-sin_da0__ + sin_da1__, cos_da0__ + cos_da1__]
    # angle change in x, positive sign is right-to-left, so only sin_da0__ is sign-reversed
    '''
    sin(-θ) = -sin(θ), cos(-θ) = cos(θ): 
    sin(da) = -sin(-da), cos(da) = cos(-da) => (sin(-da), cos(-da)) = (-sin(da), cos(da))
    '''
    ga__ = np.hypot( np.arctan2(*day__), np.arctan2(*dax__) )
    '''
    gradient of angle, initial deviation: ga - 90_degree, primary sign comp? 
    comp_aga val = 2nd deviation, sign-agnostic interruption | wave: expected reversion, same for d sign?
    '''

    i__ = i__[:-1, :-1]  # for summation in Dert
    g__ = g__[:-1, :-1]  # for summation in Dert
    m__ = m__[:-1, :-1]
    dy__ = dy__[:-1, :-1]  # passed on as idy
    dx__ = dx__[:-1, :-1]  # passed on as idx

    ## temporary section for debug purpose on comp_aga, comp_aga_ga
#    dert__ga = dcopy((i__, dy__, dx__, g__, m__, day__, dax__, ga__, ma__))
#    dert__aga, mask__aga = comp_aga(dcopy(dert__ga),ave=ave) # comp_aga
#    dert__aga_ga, mask__aga_ga = comp_aga(dcopy(dert__aga),ave=ave) # comp_aga_ga


    return (i__, dy__, dx__, g__, m__, day__, dax__, ga__, ma__), majority_mask


def comp_aga(dert__, ave, mask=None):  # prior fork is comp_a, cross-comp of angle in 2x2 kernels

    if mask is not None:
        majority_mask = (mask[:-1, :-1].astype(int) +
                         mask[:-1, 1:].astype(int) +
                         mask[1:, 1:].astype(int) +
                         mask[1:, :-1].astype(int)
                         ) > 1
    else:
        majority_mask = None

    i__ = [dert__[1],dert__[2]] # i = [dy,dx]
    dy__, dx__, g__, m__ = dert__[5:]  # day__,dax__,ga__,ma__ are recomputed
    g__ = nested(g__, replace_zero_nested)  # to avoid / 0

    # compute g + ave
    day__ = calc_a(dy__, g__, ave)  # sin, restore g to abs
    dax__ = calc_a(dx__, g__, ave)  # cos, restore g to abs
    a__ = [day__, dax__]

    # shift directions
    a__topleft = nested(dcopy(a__), shift_topleft) # use deep copy to create a new memory location
    a__topright = nested(dcopy(a__), shift_topright)
    a__botright = nested(dcopy(a__), shift_botright)
    a__botleft = nested(dcopy(a__), shift_botleft)

    # diagonal angle differences:
    sin_da0__, cos_da0__ = nested2(a__topleft, a__botright, angle_diff)
    sin_da1__, cos_da1__ = nested2(a__topright, a__botleft, angle_diff)

    ma1__ = nested2(sin_da0__, cos_da0__, hypot_add1_nested)
    ma2__ = nested2(sin_da0__, cos_da0__, hypot_add1_nested)
    ma__ = nested2(ma1__, ma2__, add_nested)
    # ma = inverse angle match = SAD: covert sin and cos da to 0->2 range

    # negative nested sin_da0
    sin_da0_nested__ = nested(dcopy(sin_da0__), negative_nested)

    # day__ = (-sin_da0__ - sin_da1__), (cos_da0__ + cos_da1__)
    day__ = [nested2(sin_da0_nested__, cos_da0__, subtract_nested),
             nested2(cos_da0__, cos_da1__, add_nested)]
    # angle change in y, sines are sign-reversed because da0 and da1 are top-down, no reversal in cosines

    # dax__ = (-sin_da0__ + sin_da1__), (cos_da0__ + cos_da1__)
    dax__ = [nested2(sin_da0_nested__, cos_da0__, add_nested),
             nested2(cos_da0__, cos_da1__, add_nested)]
    # angle change in x, positive sign is right-to-left, so only sin_da0__ is sign-reversed

    # np.arctan2(*day__)
    arctan_day__ = nested2(day__[0], day__[1], arctan2_nested)
    # np.arctan2(*dax__)
    arctan_dax__ = nested2(dax__[0], dax__[1], arctan2_nested)

    # ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))
    ga__ = nested2(arctan_day__, arctan_dax__, hypot_nested)
    # angle gradient, a scalar evaluated for comp_aga

    i__ = nested(i__, shift_topleft)  # for summation in Dert
    g__ = nested(g__, shift_topleft)  # for summation in Dert
    m__ = nested(m__, shift_topleft)
    dy__ = nested(dy__, shift_topleft)  # passed on as idy
    dx__ = nested(dx__, shift_topleft)  # passed on as idy

    return (i__, dy__, dx__, g__, m__, day__, dax__, ga__, ma__), majority_mask

# -----------------------------------------------------------------------------
# Utilities

def angle_diff(a2, a1):  # compare angle_1 to angle_2

    sin_1, cos_1 = a1[:]
    sin_2, cos_2 = a2[:]

    # sine and cosine of difference between angles:

    sin_da = (cos_1 * sin_2) - (sin_1 * cos_2)
    cos_da = (cos_1 * cos_2) + (sin_1 * sin_2)

    return [sin_da, cos_da]