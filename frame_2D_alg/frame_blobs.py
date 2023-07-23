'''
    2D version of first-level core algorithm includes frame_blobs, intra_blob (recursive search within blobs), and blob2_P_blob.
    -
    Blob is 2D pattern: connectivity cluster defined by the sign of gradient deviation. Gradient represents 2D variation
    per pixel. It is used as inverse measure of partial match (predictive value) because direct match (min intensity)
    is not meaningful in vision. Intensity of reflected light doesn't correlate with predictive value of observed object
    (predictive value is physical density, hardness, inertia that represent resistance to change in positional parameters)
    -
    Comparison range is fixed for each layer of search, to enable encoding of input pose parameters: coordinates, dimensions,
    orientation. These params are essential because value of prediction = precision of what * precision of where.
    Clustering here is nearest-neighbor only, same as image segmentation, to avoid overlap among blobs.
    -
    Main functions:
    - comp_axis:
    Comparison between diagonal pixels in 2x2 kernels of image forms derts: tuples of pixel + derivatives per kernel.
    The output is der__t: 2D array of pixel-mapped derts.
    - frame_blobs_root:
    Flood-fill segmentation of image der__t into blobs: contiguous areas of positive | negative deviation of gradient per kernel.
    Each blob is parameterized with summed params of constituent derts, derived by pixel cross-comparison (cross-correlation).
    These params represent predictive value per pixel, so they are also predictive on a blob level,
    thus should be cross-compared between blobs on the next level of search.
    - assign_adjacents:
    Each blob is assigned internal and external sets of opposite-sign blobs it is connected to.
    Frame_blobs is a root function for all deeper processing in 2D alg.
    -
    Please see illustrations:
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/blob_params.drawio
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/frame_blobs.png
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/frame_blobs_intra_blob.drawio
'''

import sys
from typing import Union

import numpy as np
from time import time
from collections import deque, namedtuple
from visualization.draw_frame_blobs import visualize_blobs
from class_cluster import ClusterStructure, init_param as z
from utils import kernel_slice_3x3 as ks    # use in comp_pixel

# hyper-parameters, set as a guess, latter adjusted by feedback:
ave = 30  # base filter, directly used for comp_r fork
ave_a = 1.5  # coef filter for comp_a fork
aveB = 50
aveBa = 1.5
ave_mP = 100
UNFILLED = -1
EXCLUDED_ID = -2
# FrameOfBlobs = namedtuple('FrameOfBlobs', 'I, Dy, Dx, M, blob_, der__t')
idert = namedtuple('idert', 'i, dy, d, g, ri')
adert = namedtuple('adert', 'i, g, ga, ri, dy, dx, dyy, dyx, dxy, dxx')

class CBlob(ClusterStructure):
    # comp_pixel:
    sign : bool = None
    I : float = 0.0
    Dy : float = 0.0
    Dx : float = 0.0
    G : float = 0.0
    A : float = 0.0 # blob area
    # composite params:
    M : float = 0.0 # summed PP.M, for both types of recursion?
    box : tuple = (0,0,0,0)  # y0, yn, x0, xn
    mask__ : object = None
    der__t : Union[idert, adert] = None
    der__t_roots : object = None  # map to der__t
    adj_blobs : list = z([])  # adjacent blobs
    fopen : bool = False
    # intra_blob params: # or pack in intra = lambda: Cintra
    # comp_angle:
    Dyy : float = 0.0
    Dyx : float = 0.0
    Dxy : float = 0.0
    Dxx : float = 0.0
    Ga : float = 0.0
    # derivation hierarchy:
    root_der__t : list = z([])  # from root blob, to extend der__t
    prior_forks : str = ''
    fBa : bool = False  # in root_blob: next fork is comp angle, else comp_r
    rdn : float = 1.0  # redundancy to higher blob layers, or combined?
    rng : int = 1  # comp range, set before intra_comp
    rlayers : list = z([])  # list of layers across sub_blob derivation tree, deeper layers are nested with both forks
    dlayers : list = z([])  # separate for range and angle forks per blob
    sub_blobs : list = z([[],[]])
    root : object = None  # frame or from frame_bblob
    # valt : list = z([])  # PPm_ val, PPd_ val, += M,G?
    # fsliced : bool = False  # from comp_slice
    # comp_dx:
    # Mdx : float = 0.0
    # Ddx : float = 0.0

class CEdge(ClusterStructure):  # edge blob

    # replace with directional params:
    I : float = 0.0
    Dy : float = 0.0
    Dx : float = 0.0
    G : float = 0.0
    M : float = 0.0 # summed PP.M, for both types of recursion?
    # comp_angle:
    Dyy : float = 0.0
    Dyx : float = 0.0
    Dxy : float = 0.0
    Dxx : float = 0.0
    Ga : float = 0.0
    box: tuple = (0, 0, 0, 0)  # y0, yn, x0, xn

    mask__: object = None
    der__t: Union[idert, adert] = None
    der__t_roots: object = None  # map to der__t
    adj_blobs: list = z([])  # adjacent blobs
    node_tt: list = z([[[],[]],[[],[]]])  # PP_ or G_ forks
    root: object= None  # root_: list if fork overlap
    derH : list = z([])  # formed in PPs and inherited in graphs
    aggH : list = z([[]])  # [[subH, valt, rdnt]]: cross-fork composition layers
    valt : list = z([0,0])
    rdnt : list = z([1,1])
    fback_ : list = z([])  
    
    # not sure below
    blob : object= None
    P_ : list = z([])   # we only need this in comp_slice, probably we can parse P_ into comp_slice too
    
'''
    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    1-3 letter names are normally scalars, except for P and similar classes, 
    capitalized variables are normally summed small-case variables,
    longer names are normally classes
'''

def frame_blobs_root(image, intra=False, render=False, verbose=False, use_c=False):

    if verbose: start_time = time()
    Y, X = image.shape[:2]

    id__tt = comp_axis(image)  # nested tuple of 2D arrays: [4|8 axes][i,d]: single difference per axis
    sign__ = ave - id__tt[1] > 0
    # select max d in each id__t for flood_fill(id__tt, sign__)?

    # form der__t from id__tt, for negative edge blobs only:
    der__t = comp_pixel(image)
    sign__ = ave - der__t[3] > 0   # sign is positive for below-average g in [i__, dy__, dx__, g__, ri]
    # https://en.wikipedia.org/wiki/Flood_fill:
    blob_, idmap, adj_pairs = flood_fill(der__t, sign__, prior_forks='', verbose=verbose)
    assign_adjacents(adj_pairs)  # forms adj_blobs per blob in adj_pairs
    I, Dy, Dx = 0, 0, 0
    for blob in blob_: I += blob.I; Dy += blob.Dy; Dx += blob.Dx

    frame = CBlob(I=I, Dy=Dy, Dx=Dx, der__t=der__t, rlayers=[blob_], box=(0,Y,0,X))
    # dlayers = []: no comp_a yet
    if verbose: print(f"{len(frame.rlayers[0])} blobs formed in {time() - start_time} seconds")

    if intra:  # omit for testing frame_blobs without intra_blob
        if verbose: print("\rRunning frame's intra_blob...")
        from intra_blob import intra_blob_root

        frame.rlayers += intra_blob_root(frame, render, verbose, fBa=0)  # recursive eval cross-comp range| angle| slice per blob
        if verbose: print("\rFinished intra_blob")  # print_deep_blob_forking(deep_blobs)
        # sublayers[0] is fork-specific, deeper sublayers combine sub-blobs of both forks
    '''
    if use_c:  # old version, no longer updated:
        der__t = der__t[0], np.empty(0), np.empty(0), *der__t[1:], np.empty(0)
        frame, idmap, adj_pairs = wrapped_flood_fill(der__t)
    '''
    if render: visualize_blobs(frame)
    return frame

def comp_pixel(image):

    pi__ = np.pad(image, pad_width=1, mode='edge')  # pad image with edge values
    # compute directional derivatives:
    dy__ = (
        (pi__[ks.bl] - pi__[ks.tl]) * 0.25 +            # left column
        (pi__[ks.bc] - pi__[ks.tc]) * 0.50 +            # middle column
        (pi__[ks.br] - pi__[ks.tr]) * 0.25              # right column
    )
    dx__ = (
        (pi__[ks.tr] - pi__[ks.tl]) * 0.25 +            # top row
        (pi__[ks.mr] - pi__[ks.mc]) * 0.50 *            # middle row
        (pi__[ks.br] - pi__[ks.bl]) * 0.25              # bottom row
    )
    G__ = np.hypot(dy__, dx__)                          # compute gradient magnitude
    rp__ = (                                            # mean of 3x3 kernel
        pi__[ks.tl] + pi__[ks.tc] + pi__[ks.tr] +       # top row
        pi__[ks.ml] + pi__[ks.mc] + pi__[ks.mr] +       # middle row
        pi__[ks.bl] + pi__[ks.bc] + pi__[ks.br]         # bottom row
    ) / 9
    return idert(pi__[ks.mc], dy__, dx__, G__, rp__)

def comp_pixel_2x2(image):  # 2x2 pixel cross-correlation within image, see comp_pixel_versions file for other versions and more explanation

    # input slices into sliding 2x2 kernel, each slice is a shifted 2D frame of grey-scale pixels:
    topleft__ = image[:-1, :-1]
    topright__ = image[:-1, 1:]
    bottomleft__ = image[1:, :-1]
    bottomright__ = image[1:, 1:]

    d_upright__ = bottomleft__ - topright__
    d_upleft__ = bottomright__ - topleft__

    # rotate back to 0 deg:
    dy__ = 0.5 * (d_upleft__ + d_upright__)
    dx__ = 0.5 * (d_upleft__ - d_upright__)

    G__ = np.hypot(d_upright__, d_upleft__)  # 2x2 kernel gradient (variation), match = inverse deviation, for sign_ only
    rp__ = topleft__ + topright__ + bottomleft__ + bottomright__  # sum of 4 rim pixels -> mean, not summed in blob param

    return (topleft__, dy__, dx__, G__, rp__)  # tuple of 2D arrays per param of dert (derivatives' tuple)
    # renamed der__t = (i__, dy__, dx__, g__, ri__) for readability in deeper functions
'''
    old version:
    Gy__ = ((bottomleft__ + bottomright__) - (topleft__ + topright__))  # decomposition of two diagonal differences into Gy
    Gx__ = ((topright__ + bottomright__) - (topleft__ + bottomleft__))  # decomposition of two diagonal differences into Gx
'''

def flood_fill(der__t, sign__, prior_forks, verbose=False, mask__=None, fseg=False):

    if mask__ is None: height, width = der__t[0].shape  # init der__t
    else:              height, width = mask__.shape  # intra der__t

    idmap = np.full((height, width), UNFILLED, 'int64')  # blob's id per dert, initialized UNFILLED
    if mask__ is not None:
        idmap[mask__] = EXCLUDED_ID
    if verbose:
        n_masked = 0 if mask__ is None else mask__.sum()
        step = 100 / (height * width - n_masked)  # progress % percent per pixel
        progress = 0.0; print(f"\rClustering... {round(progress)} %", end="");  sys.stdout.flush()
    blob_ = []
    adj_pairs = set()

    for y in range(height):
        for x in range(width):
            if idmap[y, x] == UNFILLED:  # ignore filled/clustered derts

                blob = CBlob(sign=sign__[y, x], root_der__t=der__t, prior_forks=prior_forks)
                blob_ += [blob]
                idmap[y, x] = blob.id
                y0, yn = y, y
                x0, xn = x, x
                # flood fill the blob, start from current position
                unfilled_derts = deque([(y, x)])
                while unfilled_derts:
                    y1, x1 = unfilled_derts.popleft()

                    # add dert to blob
                    if len(der__t) > 5: # comp_angle
                        blob.accumulate(I  = der__t[3][y1][x1],  # rp__,
                                        Dy = der__t[4][y1][x1],
                                        Dx = der__t[5][y1][x1],
                                        Dyy = der__t[6][y1][x1],
                                        Dyx = der__t[7][y1][x1],
                                        Dxy = der__t[8][y1][x1],
                                        Dxx = der__t[9][y1][x1])
                    else:  # comp_pixel or comp_range
                        blob.accumulate(I  = der__t[4][y1][x1],  # rp__,
                                        Dy = der__t[1][y1][x1],
                                        Dx = der__t[2][y1][x1])
                    blob.A += 1
                    if y1 < y0:   y0 = y1
                    elif y1 > yn: yn = y1
                    if x1 < x0:   x0 = x1
                    elif x1 > xn: xn = x1
                    # neighbors coordinates, 4 for -, 8 for +
                    if blob.sign or fseg:   # include diagonals
                        adj_dert_coords = [(y1 - 1, x1 - 1), (y1 - 1, x1),
                                           (y1 - 1, x1 + 1), (y1, x1 + 1),
                                           (y1 + 1, x1 + 1), (y1 + 1, x1),
                                           (y1 + 1, x1 - 1), (y1, x1 - 1)]
                    else:
                        adj_dert_coords = [(y1 - 1, x1), (y1, x1 + 1),
                                           (y1 + 1, x1), (y1, x1 - 1)]
                    # search neighboring derts:
                    for y2, x2 in adj_dert_coords:
                        # image boundary is reached:
                        if (y2 < 0 or y2 >= height or
                            x2 < 0 or x2 >= width or
                            idmap[y2, x2] == EXCLUDED_ID):
                            blob.fopen = True
                        # pixel is filled:
                        elif idmap[y2, x2] == UNFILLED:
                            # same-sign dert:
                            if blob.sign == sign__[y2, x2]:
                                idmap[y2, x2] = blob.id  # add blob ID to each dert
                                unfilled_derts += [(y2, x2)]
                        # else check if same-signed
                        elif blob.sign != sign__[y2, x2]:
                            adj_pairs.add((idmap[y2, x2], blob.id))  # blob.id always increases
                # terminate blob
                yn += 1; xn += 1
                blob.box = y0, yn, x0, xn
                blob.der__t = type(der__t)(
                    *(par__[y0:yn, x0:xn] for par__ in der__t))
                blob.mask__ = (idmap[y0:yn, x0:xn] != blob.id)
                blob.adj_blobs = [[],[]] # iblob.adj_blobs[0] = adj blobs, blob.adj_blobs[1] = poses
                blob.G = recompute_dert(blob.Dy, blob.Dx)
                if len(der__t) > 5:
                    blob.Ga, blob.Dyy, blob.Dyx, blob.Dxy, blob.Dxx = recompute_adert(blob.Dyy, blob.Dyx, blob.Dxy, blob.Dxx)
                if verbose:
                    progress += blob.A * step; print(f"\rClustering... {round(progress)} %", end=""); sys.stdout.flush()
    if verbose: print("\r" + " " * 79, end=""); sys.stdout.flush(); print("\r", end="")

    return blob_, idmap, adj_pairs


def recompute_dert(Dy, Dx):   # recompute params after accumulation
    return np.hypot(Dy, Dx)  # recompute G from Dy, Dx


def recompute_adert(Dyy, Dyx, Dxy, Dxx):  # recompute angle fork params after accumulation
    # normalize
    Dyy, Dyx = [Dyy, Dyx] / np.hypot(Dyy, Dyx)
    Dxy, Dxx = [Dxy, Dxx] / np.hypot(Dxy, Dxx)
    # recompute Ga
    Ga = (1 - Dyx) + (1 - Dxx)  # +1 for all positives
    return Ga, Dyy, Dyx, Dxy, Dxx


def assign_adjacents(adj_pairs):  # adjacents are connected opposite-sign blobs
    '''
    Assign adjacent blobs bilaterally according to adjacent pairs' ids in blob_binder.
    '''
    for blob_id1, blob_id2 in adj_pairs:
        blob1 = CBlob.get_instance(blob_id1)
        blob2 = CBlob.get_instance(blob_id2)

        y01, yn1, x01, xn1 = blob1.box
        y02, yn2, x02, xn2 = blob2.box

        if blob1.fopen and blob2.fopen:
            pose1 = pose2 = 2
        elif y01 < y02 and x01 < x02 and yn1 > yn2 and xn1 > xn2:
            pose1, pose2 = 0, 1  # 0: internal, 1: external
        elif y01 > y02 and x01 > x02 and yn1 < yn2 and xn1 < xn2:
            pose1, pose2 = 1, 0  # 1: external, 0: internal
        else:
            if blob2.A > blob1.A:
                pose1, pose2 = 0, 1  # 0: internal, 1: external
            else:
                pose1, pose2 = 1, 0  # 1: external, 0: internal
        # bilateral assignments
        '''
        if f_segment_by_direction:  # pose is not needed
            blob1.adj_blobs += [blob2]
            blob2.adj_blobs += [blob1]
        '''
        blob1.adj_blobs[0] += [blob2]
        blob1.adj_blobs[1] += [pose2]
        blob2.adj_blobs[0] += [blob1]
        blob2.adj_blobs[1] += [pose1]


if __name__ == "__main__":
    import argparse
    from utils import imread
    # Parse arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//toucan_small.jpg')
    argument_parser.add_argument('-v', '--verbose', help='print details, useful for debugging', type=int, default=1)
    argument_parser.add_argument('-r', '--render', help='render the process', type=int, default=0)
    argument_parser.add_argument('-c', '--clib', help='use C shared library', type=int, default=0)
    argument_parser.add_argument('-n', '--intra', help='run intra_blobs after frame_blobs', type=int, default=1)
    argument_parser.add_argument('-e', '--extra', help='run frame_recursive after frame_blobs', type=int, default=0)
    args = argument_parser.parse_args()
    image = imread(args.image)
    verbose = args.verbose
    intra = args.intra
    render = args.render

    start_time = time()
    if args.extra:  # not functional yet
        from frame_recursive import frame_recursive
        frame = frame_recursive(image, intra, render, verbose)
    else:
        frame = frame_blobs_root(image, intra, render, verbose)

    end_time = time() - start_time
    if args.verbose:
        print(f"\nSession ended in {end_time:.2} seconds", end="")
    else:
        print(end_time)

    '''
    Test fopen:
        if args.verbose:
        for i, blob in enumerate(frame.blob_):
        # check if fopen is correct
            # if fopen, y0 = 0, or x0 = 0, or yn = frame's y size or xn = frame's x size
            if blob.box[0] == 0 or blob.box[2] == 0 or blob.box[1] == blob.root_der__t[0].shape[0] or blob.box[3] == blob.root_der__t[0].shape[1]:
                if not blob.fopen: # fopen should be true when blob touches the frame boundaries
                    print('fopen is wrong on blob '+str(i))
    '''