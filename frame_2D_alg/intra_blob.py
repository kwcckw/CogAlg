'''
    intra_blob recursively evaluates each blob for three forks of extended internal cross-comparison and sub-clustering:

    - comp_r: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - comp_a: angle cross-comp in high-variation edge areas of positive deviation of gradient, forming gradient of angle,
    - P_blobs forms roughly edge-orthogonal Ps, evaluated for comp_d, then evaluates their stacks for comp_P

    Please see diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_scheme.png

    Blob structure, for all layers of blob hierarchy:
    root_dert__,
    Dert = S, Ly, I, Dy, Dx, G, M, Day, Dax, Ga, Ma
    # S: area, Ly: vertical dimension
    # I: input; Dy, Dx: renamed Gy, Gx; G: gradient; M: match; Day, Dax, Ga, Ma: angle Dy, Dx, G, M
    sign,
    box,  # y0, yn, x0, xn
    dert__,  # box of derts, each = i, dy, dx, g, m, day, dax, ga, ma
    # next fork:
    fia,  # flag: input is from comp angle
    fca,  # flag: current fork is comp angle
    rdn,  # redundancy to higher layers
    rng,  # comparison range
    sub_layers  # [sub_blobs ]: list of layers across sub_blob derivation tree
                # deeper layers are nested, multiple forks: no single set of fork params?
'''
from collections import deque, defaultdict
from frame_blobs_defs import CDeepBlob
from class_bind import AdjBinder
from frame_blobs import assign_adjacents, flood_fill
from intra_comp import comp_r, comp_a
from frame_blobs_imaging import visualize_blobs
from itertools import zip_longest
from utils import pairwise
import numpy as np
from P_blobs import cluster_derts_P

# from comp_P_draft import comp_P_blob

# filters, All *= rdn:
ave = 50  # fixed cost per dert, from average m, reflects blob definition cost, may be different for comp_a?
aveB = 50  # fixed cost per intra_blob comp and clustering


# --------------------------------------------------------------------------------------------------------------
# functions, ALL WORK-IN-PROGRESS:

def intra_blob(blob, **kwargs):  # recursive input rng+ | angle cross-comp within blob

    Ave = int(ave * blob.rdn); AveB = int(aveB * blob.rdn)

    if kwargs.get('render') is not None:  # stop rendering sub-blobs when blob is too small
        if blob.S < 100:
            kwargs['render'] = False

    spliced_layers = []  # to extend root_blob sub_layers
    ext_dert__, ext_mask = extend_dert(blob)

    if blob.fia:  # comp_a -> P_blobs or comp_aga

        dert__, mask = comp_a(ext_dert__, ext_mask)  # -> ga sub_blobs -> P_blobs (comp_d, comp_P)
        if mask.shape[0] > 2 and mask.shape[1] > 2 and False in mask:  # min size in y and x, least one dert in dert__

            # cluster_derts_P eval, tentative, no cluster_derts_P yet:
            if blob.G * (1 - blob.Ga / (4.45 * blob.S)) - AveB > 0:  # max_ga=6? max ga = 4.4428, we use 4.45 so that max possible value is less than 1 (4.4428/4.45<1)
                # G reduced by Ga value, base G is second deviation or specific borrow value?
                # flatten day and dax, not generalized for nested day and dax yet:
                dert__ = list(dert__)
                dert__ = (dert__[0], dert__[1], dert__[2], dert__[3], dert__[4],
                          dert__[5][0], dert__[5][1], dert__[6][0], dert__[6][1],
                          dert__[7], dert__[8])

                crit__ = dert__[3] * (1 - dert__[7] / 4.45) - Ave  # max_ga=6, record separately from g and ga?
                # ga is not signed, thus additional eval, different Ave?
                blob.fca=0
                sub_eval(blob, dert__, crit__, mask, **kwargs)

            # comp_aga eval, inverse relative ga value, tentative, no comp_aga yet:
            elif blob.G / (1 - blob.Ga / (4.45 * blob.S)) - AveB > 0:  # max_ga=6, init G is 2nd deviation or specific borrow value?
                # G increased by Ga value,  flatten day and dax?
                crit__ = dert__[3] / (1 - dert__[7] / 4.45) - Ave  # ~ eval per blob, record separately from g and ga?
                # ga is not signed, thus additional eval, different Ave?
                blob.fca = 1
                sub_eval(blob, dert__, crit__, mask, **kwargs)

            spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                              zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

    else:  # comp_r -> comp_r or comp_a
        if blob.M > AveB:
            dert__, mask = comp_r(ext_dert__, Ave, blob.fia, ext_mask)
            crit__ = dert__[4]  # signed inverse deviation of SAD

            if mask.shape[0] > 2 and mask.shape[1] > 2 and False in mask:  # min size in y and x, least one dert in dert__
                sub_eval(blob, dert__, crit__, mask, **kwargs)
                spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                                  zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

        elif blob.G > AveB:
            dert__, mask = comp_a(ext_dert__, ext_mask)  # -> m sub_blobs
            crit__ = dert__[3]  # signed deviation of g

            if mask.shape[0] > 2 and mask.shape[1] > 2 and False in mask:  # min size in y and x, least one dert in dert__
                sub_eval(blob, dert__, crit__, mask, **kwargs)
                spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                                  zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

    return spliced_layers


def sub_eval(blob, dert__, crit__, mask, **kwargs):
    Ave = ave * blob.rdn;  AveB = aveB * blob.rdn

    if blob.fia and not blob.fca:  # cluster_dert_P -> terminate fork

        sub_frame = cluster_derts_P(dert__, mask, crit__, Ave)
        sub_blobs = sub_frame['blob__']
        blob.Ls = len(sub_blobs)  # for visibility and next-fork rd
        blob.sub_layers = [sub_blobs]  # 1st layer of sub_blobs
        print('dert_P fork')

    else:  # comp_r, comp_a and comp_aga
        sub_blobs = cluster_derts(dert__, crit__, mask, verbose=False, **kwargs)  # -> m sub_blobs
        blob.Ls = len(sub_blobs)  # for visibility and next-fork rdn
        blob.sub_layers = [sub_blobs]  # 1st layer of sub_blobs

        for sub_blob in sub_blobs:  # evaluate for intra_blob comp_a | comp_r | P_blobs:
            G = blob.G
            adj_G = blob.adj_blobs[2]
            if blob.fia:
                # comp_aga
                borrow = min(abs(G), abs(adj_G) / 2)  # or adjacent M if negative sign?
                if sub_blob.G + borrow > AveB:  # also if +Ga, +Gaga, +Gr, +Gagr, +Grr...
                    # comp_aga: runnable but not correct, the issue of nested day and dax need to be fixed first
                    sub_blob.fia = 1
                    sub_blob.rdn = sub_blob.rdn + 1 + 1 / blob.Ls
                    blob.sub_layers += intra_blob(sub_blob, **kwargs)
                    print('aga fork')

            else:  # comp_r or comp_a
                borrow = min(abs(G), abs(adj_G) / 2)  # or adjacent M if negative sign?

                if min(sub_blob.G, borrow) > AveB:  # also if +Ga, +Gaga, +Gr, +Gagr, +Grr...
                    # comp_a:
                    sub_blob.rdn = sub_blob.rdn + 1 + 1 / blob.Ls
                    sub_blob.fia = 1
                    blob.sub_layers += intra_blob(sub_blob, **kwargs)
                    print('a fork')
                elif sub_blob.M > AveB:  # if +Mr, +Mrr...
                    # comp_r:
                    sub_blob.rdn = sub_blob.rdn + 1 + 1 / blob.Ls
                    sub_blob.fia = 0
                    sub_blob.rng = blob.rng * 2
                    blob.sub_layers += intra_blob(sub_blob, **kwargs)
                    print('r fork')


def cluster_derts(dert__, crit__, mask, verbose=False, **kwargs):
    # this is function should be folded into flood_fill()

    ''' obsolete define clustering criterion:
    if fia:      # input is from comp_a
        if fca:  # comp_a eval by g / cos(ga)
            crit__ = dert__[3] / np.cos(dert__[7]) - Ave  # combined value, no comp_r: no use for ma?
        else:    # P_blobs eval by g * cos(ga)
            crit__ = dert__[3] * np.cos(dert__[7]) - Ave  # separate from g and ga
    else:        # input is from comp_r
        if fca:  # comp_a eval by g
            crit__ = dert__[3] - Ave
        else:    # comp_r eval by m
            crit__ = dert__[4] - Ave
    '''
    if kwargs.get('use_c'):
        raise NotImplementedError
        (_, _, _, blob_, _), idmap, adj_pairs = flood_fill()
    else:
        blob_, idmap, adj_pairs = flood_fill(dert__,
                                             sign__=crit__ > 0,
                                             verbose=verbose,
                                             mask=mask,
                                             blob_cls=CDeepBlob,
                                             accum_func=accum_blob_Dert)

    assign_adjacents(adj_pairs, CDeepBlob)
    if kwargs.get('render', False):
        visualize_blobs(idmap, blob_,
                        winname=f"Deep blobs (fcr = {fcr}, fia = {fia})")

    return blob_


def extend_dert(blob):  # extend dert borders (+1 dert to boundaries)

    y0, yn, x0, xn = blob.box  # extend dert box:
    rY, rX = blob.root_dert__[0].shape  # higher dert size

    # determine pad size
    y0e = max(0, y0 - 1)
    yne = min(rY, yn + 1)
    x0e = max(0, x0 - 1)
    xne = min(rX, xn + 1)  # e is for extended

    # take ext_dert__ from part of root_dert__
    ext_dert__ = []
    for derts in blob.root_dert__:
        if derts is not None:
            if type(derts) == tuple:  # tuple of 2 for day, dax - (Dyy, Dyx) or (Dxy, Dxx)
                ext_dert__.append(derts[0][y0e:yne, x0e:xne])
                ext_dert__.append(derts[1][y0e:yne, x0e:xne])
            else:
                ext_dert__.append(derts[y0e:yne, x0e:xne])
        else:
            ext_dert__.append(None)
    ext_dert__ = tuple(ext_dert__)  # change list to tuple

    # extended mask
    ext_mask = np.pad(blob.mask,
                      ((y0 - y0e, yne - yn),
                       (x0 - x0e, xne - xn)),
                      constant_values=True, mode='constant')

    return ext_dert__, ext_mask


def accum_blob_Dert(blob, dert__, y, x):
    blob.I += dert__[0][y, x]
    blob.Dy += dert__[1][y, x]
    blob.Dx += dert__[2][y, x]
    blob.G += dert__[3][y, x]
    blob.M += dert__[4][y, x]

    if len(dert__) > 5:  # past comp_a fork
        blob.Dyy += dert__[5][0][y, x]
        blob.Dyx += dert__[5][1][y, x]
        blob.Dxy += dert__[6][0][y, x]
        blob.Dxx += dert__[6][1][y, x]
        blob.Ga += dert__[7][y, x]
        blob.Ma += dert__[8][y, x]
        
        
        
        
# calculation to get max ga
'''
1. From frame_blobs' comp_pixel
Given:
Gy__ = ((bottomleft__ + bottomright__) - (topleft__ + topright__))
Gx__ = ((topright__ + bottomright__) - (topleft__ + bottomleft__))  # same as decomposition of two diagonal differences into Gx

Calculation：
max of gy or gx = (255+255)-(0+0) = 510
min of gy or gx = (0+0)-(255+255) = -510

2. From frame_blobs' comp_pixel
Given:
G__ = (np.hypot(Gy__, Gx__) - ave).astype('int')
ave = 30

Calculations:
max of g = sqrt( (510^2) + (510^2) ) = 721.24891681 - ave = 691.24891681
min of g = sqrt( (0^2) + (0^2) ) = 0 - ave = -30

3. From intra_comp's comp_a
Given:
a__ = [gy__, gx__] / g__

Calculations:
max of a = 510 / 1 = 510 
min of a = -510 / 1  = -510

4.
Given:
sin_da0__, cos_da0__ = angle_diff(a1, a2)
sin_da1__, cos_da1__ = angle_diff(a1, a2)
sin_1, cos_1 = a1[:]
sin_2, cos_2 = a2[:]
sin_da = (cos_1 * sin_2) - (sin_1 * cos_2)
cos_da = (cos_1 * cos_2) + (sin_1 * sin_2)

Calculations:
max of sin_da or cos_da =  (510*510)-(-510*510) = 520200
min of sin_da or cos_da = (-510*510)+(-510*510) = -520200

5. 
Given:
day__ = (-sin_da0__ - sin_da1__), (cos_da0__ + cos_da1__)
dax__ = (-sin_da0__ + sin_da1__), (cos_da0__ + cos_da1__)

Calculations:
max of day or dax = (-520200-(-520200)) , (520200+520200) = (1040400,1040400)
min of day or dax = (-(520200)+(-520200)) , ((-520200)+(-520200)) = (-1040400,-1040400)

6. 
Given:
np.arctan2(*day__)
np.arctan2(*dax__)

Calculations:
max of np.arctan2(day or dax) = np.arctan2(0,-1040400) = 3.141592653589793
min of np.arctan2(day or dax) = np.arctan2(-1040400,-1040400) = -2.356194490192345

7. 
Given:
ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))

Calculations:
max of ga = sqrt( (3.141592653589793^2) + (3.141592653589793^2) ) = 4.442882938158366
min of ga = sqrt( (0^2) + (0^2) ) = 0

'''
