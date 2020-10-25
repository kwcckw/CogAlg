"""
Cross-comparison of pixels or gradients, in 2x2 or 3x3 kernels
"""

import numpy as np
import functools
from copy import deepcopy as dcopy

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


def comp_a(dert__, fga, ave, mask=None):  # cross-comp of angle in 2x2 kernels

    if mask is not None:
        majority_mask = (mask[:-1, :-1].astype(int) +
                         mask[:-1, 1:].astype(int) +
                         mask[1:, 1:].astype(int) +
                         mask[1:, :-1].astype(int)
                         ) > 1
    else:
        majority_mask = None


    if fga: # prior fork is comp_a
        i__ = dert__[0]
        dy__, dx__, g__, m__  = dert__[5:]
    else: # prior fork is not comp_a
        i__, dy__, dx__, g__, m__ = dert__[:5]  # day__,dax__,ga__,ma__ are recomputed

    # to avoid / 0
    g__ = nested_process(g__, nested_replace_zero)
    
    # compute g + ave
    
    
    ady__ = nested_compute_a(dy__, g__, ave) # angle, restore g to abs, similar to calc_a
    adx__ = nested_compute_a(dx__, g__, ave) # angle, restore g to abs, similar to calc_a
    a__ = [ady__,adx__]
    
    # shift directions
    a__topleft = nested_process(dcopy(a__), shift_topleft)
    a__topright = nested_process(dcopy(a__), shift_topright)
    a__botright = nested_process(dcopy(a__), shift_botright)
    a__botleft = nested_process(dcopy(a__), shift_botleft)

    # diagonal angle differences:
    sin_da0__, cos_da0__ = nested_process2(a__topleft,a__botright,nested_angle_diff)
    sin_da1__, cos_da1__ = nested_process2(a__topright, a__botleft,nested_angle_diff)

    ma1__ = nested_process2(sin_da0__, cos_da0__, nested_hypot_add_1)
    ma2__ = nested_process2(sin_da0__, cos_da0__, nested_hypot_add_1)
    ma__ = nested_process2(ma1__,ma2__,nested_add)
    # ma = inverse angle match = SAD: covert sin and cos da to 0->2 range

    # negative nested sin_da0
    n_sin_da0__ = nested_process(dcopy(sin_da0__), nested_negative)

    # day__ = (-sin_da0__ - sin_da1__), (cos_da0__ + cos_da1__)
    day__ = [nested_process2(n_sin_da0__, cos_da0__, nested_subtract),\
             nested_process2(cos_da0__, cos_da1__, nested_add)]
    # angle change in y, sines are sign-reversed because da0 and da1 are top-down, no reversal in cosines

    # dax__ = (-sin_da0__ + sin_da1__), (cos_da0__ + cos_da1__)
    dax__ = [nested_process2(n_sin_da0__, cos_da0__, nested_add),\
             nested_process2(cos_da0__, cos_da1__, nested_add)]
    # angle change in x, positive sign is right-to-left, so only sin_da0__ is sign-reversed
    
    '''
    sin(-θ) = -sin(θ), cos(-θ) = cos(θ): 
    sin(da) = -sin(-da), cos(da) = cos(-da) => (sin(-da), cos(-da)) = (-sin(da), cos(da))
    '''
    
    # np.arctan2(*day__)
    arctan_day__ = nested_process2(day__[0], day__[1], nested_arctan2)
    # np.arctan2(*dax__)
    arctan_dax__ = nested_process2(dax__[0], dax__[1], nested_arctan2)
    
    #ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))
    ga__ = nested_process2(arctan_day__, arctan_dax__, nested_hypot)   
    # angle gradient, a scalar evaluated for comp_aga

    i__ = nested_process(i__, shift_topleft) # for summation in Dert
    g__ = nested_process(g__, shift_topleft) # for summation in Dert
    m__ = nested_process(m__, shift_topleft) 
    dy__ = nested_process(dy__, shift_topleft) # passed on as idy
    dx__ = nested_process(dx__, shift_topleft) # passed on as idy
    
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

# -----------------------------------------------------------------------------
# Utilities for nested operations
    
def nested_process(element__,process_function,*args):
    '''
    nested operation on 1 variable based on the provided function
    '''
    if isinstance(element__ ,list):
        if len(element__)>1 and isinstance(element__[0],list):
            for i, element_ in enumerate(element__):
                element__[i] = nested_process(element_,process_function,*args)
        else:
            element__ = process_function(element__,*args)
    else:
        element__ = process_function(element__,*args)
    return element__

def nested_process2(element1__,element2__,process_function):
    '''
    nested operation on 2 variables based on the provided function
    '''
    element__ = dcopy(element1__)
    if isinstance(element1__[0],list):
        for i, (element1_,element2_) in enumerate(zip(element1__,element2__)):
                element__[i] = nested_process2(element1_,element2_, process_function)
    else:
        element__ = process_function(element1__,element2__)
    return element__



def nested_compute_a(element1__, element2__, ave):
    '''
    nested operation to compute a from gy,gx, g and ave
    '''
    element__ = dcopy(element1__)
    if isinstance(element2__[0],list):
        for i, (element1_,element2_) in enumerate(zip(element1__,element2__)):
                element__[i] = nested_compute_a(element1_,element2_, ave)
    else:
        if isinstance(element2__,list):
            for i, (element1_,element2_) in enumerate(zip(element1__,element2__)):
                element__[i] = [element1_[0]/element2_,element1_[1]/element2_]
        else:
                element__ = [element1__[0]/element2__,element1__[1]/element2__]

    return element__

def shift_topleft(element_):
    '''
    shift variable in top left direction
    '''
    if isinstance(element_,list):
        for i, element in enumerate(element_):
            element_[i] = element[:-1, :-1]
    else:
        element_ = element_[:-1, :-1]
    return element_

def shift_topright(element_):
    '''
    shift variable in top right direction
    '''
    if isinstance(element_,list):
        for i, element in enumerate(element_):
            element_[i] = element[:-1, 1:]
    else:
        element_ = element_[:-1, 1:]
    return element_

def shift_botright(element_):
    '''
    shift variable in bottom right direction
    '''
    if isinstance(element_,list):
        for i, element in enumerate(element_):
            element_[i] = element[1:, 1:]
    else:
        element_ = element_[1:, 1:]
    return element_

def shift_botleft(element_):
    '''
    shift variable in bottom left direction
    '''
    if isinstance(element_,list):
        for i, element in enumerate(element_):
            element_[i] = element[1:, :-1]
    else:
        element_ = element_[1:, :-1]
    return element_

def nested_negative(element_):
    '''
    complement all values in the variable
    '''
    if isinstance(element_,list):
        for i, element in enumerate(element_):
            element_[i] = -element
    else:
        element_ = -element_   
    return element_

def nested_replace_zero(element_):
    '''
    replace all 0 values in the variable with 1
    '''
    if isinstance(element_,list):
        for i, element in enumerate(element_):
            element[np.where(element == 0)] = 1
            element_[i] = element
    else:
        element_[np.where(element_ == 0)] = 1
    return element_
    

def nested_angle_diff(a2, a1):  
    '''
    compare angle_1 to angle_2
    '''
    
    sin_1, cos_1 = a1[:]
    sin_2, cos_2 = a2[:]

    # sine and cosine of difference between angles:

    sin_da = (cos_1 * sin_2) - (sin_1 * cos_2)
    cos_da = (cos_1 * cos_2) + (sin_1 * sin_2)

    return [sin_da, cos_da]

def nested_hypot(element1,element2):
    '''
    hypot of 2 elements
    '''
    return [np.hypot(element1[0], element2[0]),np.hypot(element1[1], element2[1])]

def nested_hypot_add_1(element1,element2):
    '''
    hypot of 2 (elements+1)
    '''
    return [np.hypot(element1[0] + 1, element2[0] + 1),np.hypot(element1[1] + 1, element2[1] + 1)]

def nested_add(element1,element2):
    '''
    sum of 2 variables
    '''
    return [element1[0] + element2[0] , element1[1] + element2[1]]

def nested_subtract(element1,element2):
    '''
    difference of 2 variables
    '''
    return [element1[0] - element2[0] , element1[1] - element2[1]]

def nested_arctan2(element1,element2):
    '''
    arc tan of 2 variables
    '''
    return [np.arctan2(element1[0], element2[0]), np.arctan2(element1[1], element2[1])]


'''

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


    ma__ = np.hypot(sin_da0__ + 1, cos_da0__ + 1) + np.hypot(sin_da1__ + 1, cos_da1__ + 1)
    # ma = inverse angle match = SAD: covert sin and cos da to 0->2 range

    day__ = (-sin_da0__ - sin_da1__), (cos_da0__ + cos_da1__)
    # angle change in y, sines are sign-reversed because da0 and da1 are top-down, no reversal in cosines

    dax__ = (-sin_da0__ + sin_da1__), (cos_da0__ + cos_da1__)
    # angle change in x, positive sign is right-to-left, so only sin_da0__ is sign-reversed
    
    # sin(-θ) = -sin(θ), cos(-θ) = cos(θ): 
    # sin(da) = -sin(-da), cos(da) = cos(-da) => (sin(-da), cos(-da)) = (-sin(da), cos(da))
    
    ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))
    # angle gradient, a scalar evaluated for comp_aga

    i__ = i__[:-1, :-1]  # for summation in Dert
    g__ = g__[:-1, :-1]  # for summation in Dert
    m__ = m__[:-1, :-1]
    dy__ = dy__[:-1, :-1]  # passed on as idy
    dx__ = dx__[:-1, :-1]  # passed on as idx

    ## temporary section for debug purpose
    dert__ga, mask__ga = comp_a_nested(dert__,fga=0,ave=ave) # comp_a
    dert__aga, mask__aga = comp_a_nested(dert__ga,fga=1,ave=ave) # comp_aga
    dert__aga_ga, mask__aga_ga = comp_a_nested(dert__aga,fga=1,ave=ave) # comp_aga_ga

    return (i__, dy__, dx__, g__, m__, day__, dax__, ga__, ma__), majority_mask

'''


