"""
Cross-comparison of pixels, angles, or gradients, in 2x2 or 3x3 kernels
"""

import numpy as np
import numpy.ma as ma

# -----------------------------------------------------------------------------
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

def comp_g_draft(dert__):
    """
    Cross-comp of g or ga in 2x2 kernels
    Parameters
    ----------
    dert__ : array-like, each dert = (i, g, dy, dx, da0, da1, da2, da3)
    Returns
    -------
    gdert__ : masked_array
        Output dert = (g, gg, gdy, gdx, gm, ga, day, dax).
    Examples
    --------
    >>> # actual python console code
    >>> dert__ = 'specific value'
    >>> comp_g(dert__)
    'specific output'
    Notes
    -----
    Comparand is dert[1]
    """
    g__, dy__, dx__ = dert__[1:4]
    dat__ = dert__[-1]  # tuples of four da s, computed in prior comp_a

    # please check, this is my draft, almost certainly wrong:

    day = sum([dat__[i][0] * Y_COEFFS[0][i] for i in range(len(dat__))])
    dax = sum([dat__[i][1] * X_COEFFS[0][i] for i in range(len(dat__))])

    ga = np.hypot(day, dax)

    '''
    a1__ = a__[:, :-1, :-1]                         # top left
    a2__ = a__[:, :-1, 1:]                          # top right
    a3__ = a__[:, 1:, 1:]                           # bottom right
    a4__ = a__[:, 1:, :-1]                          # bottom left
    a_central = ((a1__ + a2__ + a3__ + a4__) / 4)   # central

    # angle difference
    da__ = np.stack((angle_diff(a_central, a__1),
                     angle_diff(a_central, a__2),
                     angle_diff(a_central, a__3),
                     angle_diff(a_central, a__4)))
    '''
    # 2x2 cross-comp DRAFT:

    g0__ = g__[:-1, :-1]
    g1__ = g__[:-1, 1:]
    g2__ = g__[1:, 1:]
    g3__ = g__[1:, :-1]

    # compute gg from dg s = g - _g*cos(da)
    # match = min(g, _g*cos(da))
    # gm = ma.array(np.ones())

    # pack gdert
    gdert = ma.stack(g, gg, gdy, gdx, gm, day, dax)

    return gdert


def comp_r_draft(dert__, fig):
    """
    Cross-comparison of input param (dert[0]) over rng passed from intra_blob.
    This fork is selective for blobs with below-average gradient,
    where input intensity didn't vary much in shorter-range cross-comparison.
    Such input is predictable enough for selective sampling: skipping current
    rim derts as kernel-central derts in following comparison kernels.
    Skipping forms increasingly sparse output dert__ for greater-range cross-comp,
    hence rng: distance between central pixels of compared derts,
    increases as 2^n, where n starts at 0:
    rng = 1: 3x3 kernel,
    rng = 2: 5x5 kernel,
    rng = 4: 9x9 kernel,
    ...
    skipping rim derts as following-comp central derts (forming sparse output dert__)
    means that configuration of input derts in the next-rng kernel will always be 3x3.
    Parameters
    ----------
    dert__ : array-like
        dert's structure is (i, g, dy, dx, m, ?(idy, idx)).
    fig : bool
        True if input is g.
    Returns
    -------
    rdert__ : masked_array
        Output's structure is (i, g, dy, dx, m, ?(idy, idx)).
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

    # input is gdert (g,  gg, gdy, gdx, gm, iga, iday, idax)
    # input is dert  (i,  g,  dy,  dx,  m)  or
    # input is rdert (ir, gr, dry, drx, mr)
    i__, g__, dy__, dx__, m__ = dert__[0:5]

    # get sparsed value
    ri__   = i__[1::2,1::2]
    rg__   = g__[1::2,1::2]
    rdy__  = dy__[1::2,1::2]
    rdx__  = dx__[1::2,1::2]
    rm__   = m__[1::2,1::2]

    # get each direction
    ri__topleft      = ri__[:-2,:-2]
    ri__top          = ri__[:-2,1:-1]
    ri__topright     = ri__[:-2,2:]
    ri__right        = ri__[1:-1,2:]
    ri__bottomright  = ri__[2:,2:]
    ri__bottom       = ri__[2:,1:-1]
    ri__bottomleft   = ri__[2:,:-2]
    ri__left         = ri__[1:-1,:-2]

    if fig:  # input is g
        # OUTDATED:

        # compute diagonal differences for g in 3x3 kernel
        drg__ = np.stack((ri__topleft- ri__bottomright,
                          ri__top-     ri__bottom,
                          ri__topright-ri__bottomleft,
                          ri__right-   ri__left))

        # compute a from the range dy,dx and g
        a__ = [rdy__,rdx__] / rg__
        # angles per direction:
        a__topleft      = a__[:,:-2,:-2]
        a__top          = a__[:,:-2,1:-1]
        a__topright     = a__[:,:-2,2:]
        a__right        = a__[:,1:-1,2:]
        a__bottomright  = a__[:,2:,2:]
        a__bottom       = a__[:,2:,1:-1]
        a__bottomleft   = a__[:,2:,:-2]
        a__left         = a__[:,1:-1,:-2]

        # compute opposing diagonals difference for a in 3x3 kernel
        dra__ = np.stack((angle_diff(a__topleft, a__bottomright),
                          angle_diff(a__top,     a__bottom),
                          angle_diff(a__topright,a__bottomleft),
                          angle_diff(a__right,   a__left)))

        # g difference  = g - g * cos(da) at each opposing diagonals
        dri__ = np.stack( (drg__[0] - drg__[0] * dra__[0][1],
                           drg__[1] - drg__[1] * dra__[1][1],
                           drg__[2] - drg__[2] * dra__[2][1],
                           drg__[3] - drg__[3] * dra__[3][1]) )

    else:  # input is pixel

    #   i difference of each opossing diagonals
        dri__ = np.stack((ri__topleft-ri__bottomright,
                          ri__top-ri__bottom,
                          ri__topright-ri__bottomleft,
                          ri__right-ri__left))


    dri__ = np.rollaxis(dri__,0,3)

    # compute dry and drx
    dry__ = (dri__ * Y_COEFFS[0][0:4]).sum(axis=-1)
    drx__ = (dri__ * X_COEFFS[0][0:4]).sum(axis=-1)

    # compute gradient magnitudes
    drg__ = ma.hypot(dry__, drx__)

    # pending m computation
    drm = []

    # rdert
    rdert = dri__, drg__, dry__, drx__,drm

    return rdert


def comp_a(dert__, fga):
    """
    cross-comp of a or aga in 2x2 kernels
    Parameters
    ----------
    dert__ : array-like
        dert's structure depends on fga
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
    >>> fga = 'specific value'
    >>> comp_a(dert__, fga)
    'specific output'
    """
    # input dert = (i,  g,  dy,  dx,  m, ga, day, dax, dat(da0, da1, da2, da3))
    i__, g__,dy__,dx__,m__ = dert__[0:5]

    if fga:  # if input is adert

        ga__,day__,dax__ = dert__[5:8]
        a__ = [day__,dax__] / ga__ # similar to calc_a

    else :
        a__ = [dy__,dx__] / g__ # similar to calc_a

    if isinstance(a__, ma.masked_array):
        a__.data[a__.mask] = np.nan
        a__.mask = ma.nomask

    # each shifted a in 2x2 kernel
    a__topleft = a__[:,:-1,:-1]
    a__topright = a__[:,:-1,1:]
    a__bottomright = a__[:,1:,1:]
    a__bottomleft = a__[:,1:,:-1]


    # get angle difference of each direction
    da__ = np.stack((angle_diff(a__topleft,   a__bottomright),
                     angle_diff(a__bottomleft,a__topright),
                     angle_diff(a__topright,  a__bottomleft)))

    day__ = (
            Y_COEFFS[0][0] * angle_diff(a__topleft, a__bottomright) +
            Y_COEFFS[0][1] * angle_diff(a__topright, a__bottomleft)
    )
    dax__ = (
            X_COEFFS[0][0] * angle_diff(a__topleft, a__bottomright) +
            X_COEFFS[0][1] * angle_diff(a__topright, a__bottomleft)
    )
    # compute gradient magnitudes (how fast angles are changing)

    ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))

    # change adert to tuple as ga__,day__,dax__ would have different dimension compared to inputs
    adert__ = i__, g__,dy__,dx__,m__,ga__,day__,dax__

    return adert__



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
    sin_1 = a1[0]
    sin_2 = a2[0]

    cos_1 = a1[1]
    cos_2 = a2[1]

    # by the formulas of sine and cosine of difference of angles
    sin_da = (cos_1 * sin_2) - (sin_1 * cos_2)
    cos_da = (sin_1 * cos_1) + (sin_2 * cos_2)

    return ma.array([sin_da, cos_da])

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

from test_sets import (
comp_pixel_test_pairs,
calc_a_test_pairs,
pixels, rderts, gderts, angles)

dert__ = gderts[2]
dert__ = np.repeat(dert__,2,axis=0)
adert__ = dert__[0:8]
dert__ = dert__[0:5]

rdert__tem = np.uint8(np.random.rand(25,25)*255)
rdert__ = np.zeros((8,25,25))
rdert__[0:8,:,:] = rdert__tem

fga = 0
adert1 = comp_a(dert__, fga)
fga = 1
adert2 = comp_a(adert__, fga)
fig = 0
rdert1 = comp_r(dert__, fig)
fig = 1
rdert2 = comp_r(rdert__, fig)

# from test_sets import (
# comp_pixel_test_pairs,
# calc_a_test_pairs,
# pixels, rderts, gderts, angles)
#
# dert__ = gderts[2]
#
# fga = 1
# fig = 0
#
# adert = comp_a(dert__, fga)
#
# dert__ = np.repeat(dert__,2,axis=0)
# rdert = comp_r(dert__, fig)
