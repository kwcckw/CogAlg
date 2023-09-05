import sys
import numpy as np
from collections import namedtuple, deque, defaultdict
from itertools import product, combinations
from .classes import CEdge, CP
from .filters import ave_g, ave_dangle
from .comp_slice import comp_angle

'''
In natural images, objects look very fuzzy and frequently interrupted, only vaguely suggested by initial blobs and contours.
Potential object is proximate low-gradient (flat) blobs, with rough / thick boundary of adjacent high-gradient (edge) blobs.
These edge blobs can be dimensionality-reduced to their long axis / median line: an effective outline of adjacent flat blob.
-
Median line can be connected points that are most equidistant from other blob points, but we don't need to define it separately.
An edge is meaningful if blob slices orthogonal to median line form some sort of a pattern: match between slices along the line.
These patterns effectively vectorize representation: they represent match and change between slice parameters along the blob.
-
This process is very complex, so it must be selective. Selection should be by combined value of gradient deviation of edge blobs
and inverse gradient deviation of flat blobs. But the latter is implicit here: high-gradient areas are usually quite sparse.
A stable combination of a core flat blob with adjacent edge blobs is a potential object.
'''

Tptuple = namedtuple("Tptuple", "I Dy Dx G M Ma L")
octant = 0.3826834323650898

def slice_edge(blob, verbose=False):

    max_mask__ = max_selection(blob)  # mask of local directional maxima of dy, dx, g
    # form slices (Ps) from max_mask__ and form links by tracing max_mask__:
    blob.P_, blob.P_link_ = trace_edge(blob, max_mask__, verbose=verbose)

def max_selection(blob):

    Y, X = blob.mask__.shape
    g__ = blob.der__t.g
    # compute direction of gradient
    with np.errstate(divide='ignore', invalid='ignore'):
        sin__, cos__ = [blob.der__t.dy, blob.der__t.dx] / g__

    # round angle to one of eight directions
    up__, lft__, dwn__, rgt__ = (sin__< -octant), (cos__< -octant), (sin__> octant), (cos__> octant)
    mdly__, mdlx__ = ~(up__ | dwn__), ~(lft__ | rgt__)
    # merge in 4 bilateral axes
    axes_mask__ = [
        mdly__ & (rgt__ | lft__), (dwn__ & rgt__) | (up__ & lft__),     #  0,  45 deg
        (dwn__ | up__) & mdlx__,  (dwn__ & lft__) | (up__ & rgt__),     # 90, 135 deg
    ]
    max_mask__ = np.zeros_like(blob.mask__, dtype=bool)
    # local max from cross-comp within axis,
    # switch to max within kernel for omnidirectional sparsity
    for axis_mask__, (ydir, xdir) in zip(axes_mask__, ((0,1),(1,1),(1,0),(1,-1))):  # y,x direction per axis
        # axis AND mask:
        mask__ = axis_mask__ & blob.mask__
        y_, x_ = mask__.nonzero()
        # neighbors:
        yn1_, xn1_ = y_ + ydir, x_ + xdir
        yn2_, xn2_ = y_ - ydir, x_ - xdir
        # computed vals
        axis1_ = (0 <= yn1_) & (yn1_ < Y) & (0 <= xn1_) & (xn1_ < X)
        axis2_ = (0 <= yn2_) & (yn2_ < Y) & (0 <= xn2_) & (xn2_ < X)
        # compare values
        not_max_ = np.zeros_like(y_, dtype=bool)
        not_max_[axis1_] |= (g__[y_[axis1_], x_[axis1_]] < g__[yn1_[axis1_], xn1_[axis1_]])
        not_max_[axis2_] |= (g__[y_[axis2_], x_[axis2_]] < g__[yn2_[axis2_], xn2_[axis2_]])
        # select maxes
        mask__[y_[not_max_], x_[not_max_]] = False
        # add to max_mask__
        max_mask__ |= mask__

    return max_mask__

def trace_edge(blob, mask__, verbose=False):

    max_ = {*zip(*mask__.nonzero())}  # convert mask__ into a set of (y,x)

    if verbose:
        step = 100 / len(max_)  # progress % percent per pixel
        progress = 0.0; print(f"\rTracing max... {round(progress)} %", end="");  sys.stdout.flush()

    P_ = []
    link_ = set()
    remaining_max_ = set(max_)
    max_olps = defaultdict(list)
    while remaining_max_:  # queue of (y,x,P)s
        y, x = remaining_max_.pop()
        maxQue = deque([(y, x, None)])    # queue tp trace start with (y, x) from max_

        while maxQue:
            # initialize dert to form P
            y, x, _P = maxQue.popleft()     # pop from queue
            i = blob.i__[blob.ibox.slice()][y, x]   # get i
            dy, dx, g = blob.der__t.get_pixel(y, x) # get dy, dx, g
            ma = ave_dangle     # m is at maximum value because P direction is the same as dert gradient direction
            assert g > 0, "g must be positive"
            P = form_P(blob, CP(yx=(y, x), axis=(dy/g, dx/g), cells={(y,x)}, dert_=[(y, x, i, dy, dx, g, ma)]))
            P_ += [P]
            if _P is not None:
                link_ |= {(_P, P)}
            for olp_yx in P.cells & max_:
                max_olps[olp_yx] += [P]

            # search in max_ path
            adjacents = remaining_max_ & {*product(range(y-1,y+2), range(x-1,x+2))}   # search neighbors
            maxQue.extend(((_y, _x, P) for _y, _x in adjacents))
            remaining_max_ -= adjacents   # set difference = first set AND not both sets: https://www.scaler.com/topics/python-set-difference/
            if verbose:
                progress += step; print(f"\rTracing max... {round(progress)} %", end=""); sys.stdout.flush()

    if verbose: print("\r" + " " * 79, end=""); sys.stdout.flush(); print("\r", end="")

    # likely irrelevant:
    olp_pairs = set()
    for yx in max_olps:
        for _P, P in combinations(max_olps[yx], r=2):  # loop thru the rest
            assert _P.id < P.id
            yx_dist = np.hypot(*np.subtract(P.yx, _P.yx))   # compute distance
            axis_diff, axis_match = comp_angle(P.axis, _P.axis) # compute angle match
            if yx_dist >= 1.0 or axis_match < ave_dangle - 0.1:
                continue
            olp_pairs |= {(_P, P)}
    for merging_P_ in connected_groups(P_, olp_pairs):
        # initialize merged P
        y, x = np.mean([merging_P.yx for merging_P in merging_P_], axis=0)
        summed_axis = np.sum([merging_P.axis for merging_P in merging_P_], axis=0)
        axis = summed_axis / np.hypot(*summed_axis)
        i, dy, dx, g = interpolate2dert(blob, y, x)
        ma = ave_dangle
        merged_P = form_P(blob, CP(yx=(y,x), axis=axis, cells={(y,x)}, dert_=[(y,x,i,dy,dx,g,ma)]))

        link_ -= combinations(merging_P_, r=2)      # remove links between merged Ps, if any
        P_ = [P for P in P_ if P not in merging_P_] # remove merging_P_
        P_ += [merged_P]   # add merged P to P_

    return P_, link_

def connected_groups(nodes, edges):
    # TODO: implement connected_groups
    # https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
    return []

def form_P(blob, P):

    scan_direction(blob, P, fleft=1)  # scan left
    scan_direction(blob, P, fleft=0)  # scan right
    # init:
    _, _, I, Dy, Dx, G, Ma = map(sum, zip(*P.dert_))
    L = len(P.dert_)
    M = ave_g*L - G
    G = np.hypot(Dy, Dx)  # recompute G
    P.ptuple = Tptuple(I, Dy, Dx, G, M, Ma, L)
    P.yx = P.dert_[L//2][:2]  # new center

    return P

def scan_direction(blob, P, fleft):  # leftward or rightward from y,x

    sin,cos = _dy,_dx = P.axis  # unpack axis
    _y, _x = P.yx               # start with pivot
    r = cos*_y - sin*_x  # from P line equation: cos*y - sin*x = r = constant
    _cy,_cx = round(_y), round(_x)  # keep initial cell
    y, x = (_y-sin,_x-cos) if fleft else (_y+sin, _x+cos)  # first dert in the direction of axis

    while True:  # scan to blob boundary or angle miss

        dert = interpolate2dert(blob, y, x)
        if dert is None: break  # blob boundary
        i, dy, dx, g = dert
        cy, cx = round(y), round(x)  # nearest cell of (y, x)
        if not blob.mask__[cy, cx]: break
        if abs(cy-_cy) + abs(cx-_cx) == 2:  # mask of cell between (y,x) and (_y,_x)
            my = (_cy+cy) / 2  # midpoint cell, P axis is above, below or over it
            mx = (_cx+cx) / 2
            _my_cos = sin * mx + r  # _my*cos at mx in P, to avoid division
            my_cos = my * cos       # new cell
            if cos < 0: my_cos, _my_cos = -my_cos, -_my_cos   # reverse sign for comparison because of cos
            if abs(my_cos-_my_cos) > 1e-5:
                adj_y, adj_x = (  # deviation from P axis: above/_y>y, below/_y<y, over/_y~=y, with reversed y:
                    ((_cy, cx) if _cy < cy else (cy, _cx)) if _my_cos < my_cos else
                    ((_cy, cx) if _cy > cy else (cy, _cx)))
                if not blob.mask__[adj_y, adj_x]: break    # if the cell is masked, stop
                P.cells |= {(adj_y, adj_x)}

        mangle,dangle = comp_angle((_dy,_dx), (dy, dx))
        if mangle < 0:  # terminate P if angle miss
            break
        P.cells |= {(cy, cx)}  # add current cell to overlap
        _cy, _cx, _dy, _dx = cy, cx, dy, dx
        if fleft:
            P.dert_ = [(y,x,i,dy,dx,g,mangle)] + P.dert_  # append left
            y -= sin; x -= cos  # next y,x
        else:
            P.dert_ = P.dert_ + [(y,x,i,dy,dx,g,mangle)]  # append right
            y += sin; x += cos  # next y,x

def interpolate2dert(blob, y, x):

    Y, X = blob.mask__.shape    # boundary
    x0, y0 = int(x), int(y)     # floor
    x1, y1 = x0 + 1, y0 + 1     # ceiling
    if x0 < 0 or x1 >= X or y0 < 0 or y1 >= Y: return None  # boundary check
    kernel = [  # cell weighing by inverse distance from float y,x:
        # https://www.researchgate.net/publication/241293868_A_study_of_sub-pixel_interpolation_algorithm_in_digital_speckle_correlation_method
        (y0, x0, (y1 - y) * (x1 - x)),
        (y0, x1, (y1 - y) * (x - x0)),
        (y1, x0, (y - y0) * (x1 - x)),
        (y1, x1, (y - y0) * (x - x0))]
    ider__t = (blob.i__[blob.ibox.slice()],) + blob.der__t

    return (sum((par__[ky, kx] * dist for ky, kx, dist in kernel)) for par__ in ider__t)