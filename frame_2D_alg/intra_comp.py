"""
Cross-comparison of pixels, angles, or gradients, in 2x2 or 3x3 kernels
"""

import numpy as np
import numpy.ma as ma

# -----------------------------------------------------------------------------
# Constants


Y_COEFFS = [
    np.array([-1, -1, 1, 1]),
    np.array([-0.5, -0.5, -0.5,  0. ,  0.5,  0.5,  0.5,  0. ]),
]

X_COEFFS = [
    np.array([-1, 1, 1, -1]),
    np.array([-0.5,  0. ,  0.5,  0.5,  0.5,  0. , -0.5, -0.5]),
]

# -----------------------------------------------------------------------------
# Functions

def comp_g(dert__, fc3):
    """
    Cross-comp of g or ga, in 2x2 kernels unless root fork is comp_r: fc3=True, sparse 3x3 comp
    Parameters
    ----------
    dert__ : array-like
        The structure is (i, g, dy, dx) for dert or (ga, day, dax) for adert.
    fc3 : bool
        Initially False, set to True for comp_a and comp_g called from comp_r fork.
    Returns
    -------
    gdert__ : masked_array
        Output's structure is (g, gg, gdy, gdx, gm, ga, day, dax).
    Examples
    --------
    >>> # actual python console code
    >>> dert__ = 'specific value'
    >>> fc3 = 'specific value'
    >>> comp_g(dert__, fc3)
    'specific output'
    Notes
    -----
    Comparand is dert[1]
    """
    pass


def comp_r(dert__, fig):
    """
    Cross-comp of input param (dert[0]) over rng set in intra_blob.
    This comparison is selective for blobs with below-average gradient,
    where input intensity doesn't vary much in shorter-range cross-comparison.
    Such input is predictable enough for selective sampling: skipping
    alternating derts as a kernel-central dert at current comparison range,
    which forms increasingly sparse input dert__ for greater range cross-comp,
    while maintaining one-to-one overlap between kernels of compared derts.
    With increasingly sparse input, unilateral rng (distance between
    central derts) can only increase as 2^(n + 1), where n starts at 0:
    With increasingly sparse input, unilateral rng (distance between
    central derts) can only increase as 2^(n + 1), where n starts at 0:
    rng = 1 : 3x3 kernel, skip orthogonally alternating derts as centrals,
    rng = 2 : 5x5 kernel, skip diagonally alternating derts as centrals,
    rng = 3 : 9x9 kernel, skip orthogonally alternating derts as centrals,
    ...
    That means configuration of preserved (not skipped) derts will always be 3x3.
    Parameters
    ----------
    dert__ : array-like
        dert's structure is (i, g, dy, dx, m).
    fig : bool
        True if input is g.
    Returns
    -------
    rdert__ : masked_array
        Output's structure is (i, g, dy, dx, m).
    Examples
    --------
    >>> # actual python console code
    >>> dert__ = 'specific value'
    >>> fig = 'specific value'
    >>> comp_r(dert__, fig)
    'specific output'
    Notes
    -----
    - Results are accumulated in the input dert.
    - Comparand is dert[0].
    """
    pass


def comp_a(dert__, fga, fc3):
    """
    cross-comp of a or aga, in 2x2 kernels unless root fork is comp_r: fc3=True.
    Parameters
    ----------
    dert__ : array-like
        dert's structure depends on fga
    fc3 : bool
        Is True if root fork is comp_r.
    fga : bool
        If True, dert's structure is interpreted as:
        (g, gg, gdy, gdx, gm, iga, iday, idax)
        Otherwise it is interpreted as:
        (i, g, dy, dx, m)
    Returns
    -------
    adert : masked_array
        adert's structure is (ga, day, dax).
    Examples
    --------
    >>> # actual python console code
    >>> dert__ = 'specific value'
    >>> odd = 'specific value'
    >>> aga = 'specific value'
    >>> comp_a(dert__, fga, fc3)
    'specific output'
    """
        
    # input is rdert (ir, gr, dry, drx, mr)
    if fc3:
        gr,dry,drx = dert__[1:4,:]
        a__ = [dry,drx]/gr
        a__ = a__[:,::2,::2]  # skip odd angle`
        
    # input is gdert (g, gg, gdy, gdx, gm, iga, iday, idax)
    elif fga:
        gg,gdy,gdx = dert__[1:4,:]
        a__ = [gdy,gdx]/gg
        
#    # input is dert (i, g, dy, dx, m)
#    elif fig:
#        g,dy,dx = dert__[1:4,:]
#        a__ = [gdy,gdx]/gg
      
    # input is adert (ga, day, dax)
    else :
        ga,day,dax = dert__[0:3,:]
        a__ = [day,dax]/ga
    
    if isinstance(a__, ma.masked_array):
        a__.data[a__.mask] = np.nan
        a__.mask = ma.nomask
      
    # input is rdert,rng = 1
    if fc3:
        
        # each shifted a
        a__topleft      = a__[:,:-2,:-2]
        a__top          = a__[:,:-2,1:-1]
        a__topright     = a__[:,:-2,2:]
        a__right        = a__[:,1:-1,2:]
        a__bottomright  = a__[:,2:,2:]
        a__bottom       = a__[:,2:,1:-1]
        a__bottomleft   = a__[:,2:,:-2]
        a__left         = a__[:,1:-1,:-2]
        
        # central of a
        a__= a__[:,1:-1,1:-1]
        
        # get angle difference of each direction
        da__ = np.stack((angle_diff(a__,a__topleft),
                         angle_diff(a__,a__top),
                         angle_diff(a__,a__topright),
                         angle_diff(a__,a__right),
                         angle_diff(a__,a__bottomright),
                         angle_diff(a__,a__bottom),
                         angle_diff(a__,a__bottomleft),
                         angle_diff(a__,a__left)));
        
        da__ = np.rollaxis(da__,0,4)               
           
        # compute day and dax              
        day__ = (da__ * Y_COEFFS[1]).sum(axis=-1)
        dax__ = (da__ * X_COEFFS[1]).sum(axis=-1)
        
        # compute gradient magnitudes (how fast angles are changing)
        ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))
        
    # input is dert, gdert or adert,rng = 0
    else:
        
        # each shifted a
        a__topleft = a__[:,:-1,:-1]
        a__topright = a__[:,:-1,1:]
        a__bottomright = a__[:,1:,1:]
        a__bottomleft = a__[:,1:,:-1]
        
        # central of a
        a__ = (a__topleft + a__topright + a__bottomright + a__bottomleft)/4
        
        # get angle difference of each direction
        da__ = np.stack((angle_diff(a__,a__topleft),
                         angle_diff(a__,a__topright),
                         angle_diff(a__,a__bottomright),
                         angle_diff(a__,a__bottomleft)))
            
        da__ = np.rollaxis(da__,0,4)    
            
        # compute day and dax
        day__ = (da__ * Y_COEFFS[0]).sum(axis=-1)
        dax__ = (da__ * X_COEFFS[0]).sum(axis=-1)
        
        # compute gradient magnitudes (how fast angles are changing)
        ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))


    adert = ma.stack((ga__,*day__,*dax__))

    return adert


def calc_a(dert__):
    """
    Compute vector representation of gradient angle.
    It is done by normalizing the vector (dy, dx).
    Numpy broad-casting is a viable option when the
    first dimension of input array (dert__) separate
    different CogAlg parameter (like g, dy, dx).
    Example
    -------
    >>> dert1 = np.array([0, 5, 3, 4])
    >>> a1 = calc_a(dert1)
    >>> print(a1)
    array([0.6, 0.8])
    >>> # 45 degrees angle
    >>> dert2 = np.array([0, 450**0.5, 15, 15])
    >>> a2 = calc_a(dert2)
    >>> print(a2)
    array([0.70710678, 0.70710678])
    >>> print(np.degrees(np.arctan2(*a2)))
    45.0
    >>> # -30 (or 330) degrees angle
    >>> dert3 = np.array([0, 10, -5, 75**0.5])
    >>> a3 = calc_a(dert3)
    >>> print(a3)
    array([-0.5      ,  0.8660254])
    >>> print(np.rad2deg(np.arctan2(*a3)))
    -29.999999999999996
    """
    return dert__[[2, 3]] / dert__[1]  # np.array([dy, dx]) / g


# -----------------------------------------------------------------------------
# Utility functions

def angle_diff(a2, a1):
    """
    Return the sine(s) and cosine(s) of the angle between a2 and a1.
    Parameters
    ----------
    a1 , a2 : array-like
        Each contains sine and cosine of corresponding angle,
        in that order. For vectorized operations, sine/cosine
        dimension must be the first dimension.
    Return
    ------
    out : MaskedArray
        The first dimension is sine/cosine of the angle(s) between
        a2 and a1.
    Note
    ----
    This only works for angles in 2D space.
    """
    return ma.array([a1[1] * a2[0] - a1[0] * a2[1],
                     a1[0] * a2[0] + a1[1] * a2[1]])

    # OLD VERSION OF angle_diff
    # # Extend a1 vector(s) into basis/bases:
    # y, x = a1
    # bases = [(x, -y), (y, x)]
    # transform_mat = ma.array(bases)
    #
    # # Apply transformation:
    # da = ma.multiply(transform_mat, a2).sum(axis=1)

    # return da

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
    
    
#from test_sets import (
#comp_pixel_test_pairs,
#calc_a_test_pairs,
#pixels, rderts, gderts, angles)
#
#dert__ = gderts[2]
#fga = 0
#fc3 = 0
#comp_a(dert__, fga, fc3)