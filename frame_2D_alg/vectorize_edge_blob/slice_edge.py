import numpy as np
from collections import defaultdict
from math import atan2, cos, floor, pi
from weakref import ref
import sys
sys.path.append("..")
from frame_blobs import CBase, CH, imread   # for CP
from intra_blob import CsubFrame
from utils import box2center
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
octant = 0.3826834323650898  # radians per octant
aveG = 10  # for vectorize
ave_g = 30  # change to Ave from the root intra_blob?
ave_mL = 2
ave_dist = 3
max_dist = 15
ave_dangle = .8  # vertical difference between angles: -1->1, abs dangle: 0->1, ave_dangle = (min abs(dangle) + max abs(dangle))/2,

class CsliceEdge(CsubFrame):

    class CEdge(CsubFrame.CBlob): # replaces CBlob

        def term(blob):  # extension of CsubFrame.CBlob.term(), evaluate for vectorization right after rng+ in intra_blob
            super().term()
            if not blob.sign and blob.G > aveG * blob.root.rdn:
                blob.vectorize()

        def vectorize(blob):  # overridden in comp_slice, agg_recursion
            blob.slice_edge()

        def slice_edge(edge):
            edge.rootd = {}
            edge.P_ = [CP(edge, yx, axis) for yx, axis in edge.select_max()]
            edge.P_.sort(key=lambda P: P.yx, reverse=True)
            edge.trace()
            # del edge.rootd

        def select_max(edge):
            max_ = []
            for (y, x), (i, gy, gx, g) in edge.dert_.items():
                # check neighbors
                new_max = True
                for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1)]:
                    for _y, _x in [(y-dy, x-dx), (y+dy, x+dx)]:
                        if (_y, _x) not in edge.dert_: continue  # skip if pixel not in edge blob
                        _i, _gy, _gx, _g = edge.dert_[_y, _x]  # get g of neighbor
                        if g < _g:
                            new_max = False
                            break
                if new_max: max_ += [((y, x), (gy/g, gx/g))]
            return max_

        def trace(edge):  # fill and trace across slices
            for P in edge.P_:
                adjacent_ = [yx for yx in edge.rootd if edge.rootd[yx] is P]
                while adjacent_:
                    _y, _x = adjacent_.pop()
                    for y, x in [(_y-1,_x),(_y,_x+1),(_y+1,_x),(_y,_x-1)]:
                        try:    # if yx has _P, try to form link
                            _P = edge.rootd[y, x]
                            if _P is not P and _P not in P.link_[0]:
                                P.link_[0] += [_P]
                        except KeyError:    # if yx empty, keep tracing
                            if (y, x) not in edge.dert_: continue
                            edge.rootd[y, x] = P
                            adjacent_ += [(y, x)]
                P.link_[0] = [Clink([_P, P]) for _P in P.link_[0]]

    CBlob = CEdge

class CP(CBase):

    def __init__(P, edge, yx, axis):
        super().__init__()
        y, x = yx
        P.axis = ay, ax = axis

        pivot = i,gy,gx,g = edge.dert_[y,x]  # dert is None if (_y, _x) not in edge.dert_: return` in `interpolate2dert`
        ma = ave_dangle  # ? max value because P direction is the same as dert gradient direction
        m = ave_g - g
        pivot += ma,m

        edge.rootd[y, x] = P
        I,G,M,Ma,L,Dy,Dx = i,g,m,ma,1,gy,gx
        P.yx_, P.dert_, P.link_ = [yx], [pivot], [[]]

        # this rotation should be recursive, use P.latuple Dy,Dx to get secondary direction, no need for axis?
        for dy,dx in [(-ay,-ax),(ay,ax)]:  # scan in 2 opposite directions to add derts to P
            P.yx_.reverse(); P.dert_.reverse()
            (_y,_x), (_,_gy,_gx,*_) = yx, pivot  # start from pivot
            y,x = _y+dy, _x+dx  # 1st extension
            while True:
                # scan to blob boundary or angle miss:
                ky, kx = round(y), round(x)
                if (round(y),round(x)) not in edge.dert_: break
                try: i,gy,gx,g = interpolate2dert(edge, y, x)
                except TypeError: break  # out of bound (TypeError: cannot unpack None)

                mangle, dangle = comp_angle((_gy,_gx), (gy, gx))
                if mangle < ave_dangle: break  # terminate P if angle miss
                # update P:
                edge.rootd[ky, kx] = P
                m = ave_g - g
                I += i; Dy += dy; Dx += dx; G += g; Ma += ma; M += m; L += 1
                P.yx_ += [(y,x)]; P.dert_ += [(i,gy,gx,g,ma,m)]
                # for next loop:
                y += dy; x += dx
                _y,_x,_gy,_gx = y,x,gy,gx

        P.yx = tuple(np.mean([P.yx_[0], P.yx_[-1]], axis=0))
        P.latuple = I, G, M, Ma, L, (Dy, Dx)
        P.derH = CH()

    def __repr__(P): return f"P(id={P.id})"


class Clink(CBase):  # the product of comparison between two nodes

    def __init__(l, node_=None,rim=None, derH=None, extH=None, roott=None, distance=0, angle=None, box=None ):
        super().__init__()
        l.node_ = [] if node_ is None else node_  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.angle = [0,0] if angle is None else angle  # dy,dx between node centers
        l.distance = distance  # distance between node centers
        l.Et = [0,0,0,0]  # graph-specific, accumulated from surrounding nodes in node_connect
        l.relt = [0,0]
        l.rim = []  # for der+, list of mediating Clinks in hyperlink in roughly the same direction, as in hypergraph
        l.derH = CH() if derH is None else derH
        l.extH = CH() if extH is None else extH  # for der+
        l.roott = [None, None] if roott is None else roott  # clusters that contain this link
        l.compared_ = []
        # for rng+
        l.nest = 0
        l.n = 1  # default n
        l.area = 0
        l.box = [np.inf, np.inf, -np.inf, -np.inf] if box is None else box  # y,x,y0,x0,yn,xn
        # dir: bool  # direction of comparison if not G0,G1, only needed for comp link?
        # n: always min(node_.n)?

    def __bool__(l): return bool(l.derH.H)


def interpolate2dert(edge, y, x):
    if (y, x) in edge.dert_: return edge.dert_[y, x]  # if edge has (y, x) in it

    # get coords surrounding dert:
    y_ = [fy] = [floor(y)]; x_ = [fx] = [floor(x)]
    if y != fy: y_ += [fy+1]    # y is non-integer
    if x != fx: x_ += [fx+1]    # x is non-integer

    I, Dy, Dx, G = 0, 0, 0, 0
    K = 0
    for _y in y_:
        for _x in x_:
            if (_y, _x) not in edge.dert_: continue
            i, dy, dx, g = edge.dert_[_y, _x]
            k = (1 - abs(_y-y)) * (1 - abs(_x-x))
            I += i*k; Dy += dy*k; Dx += dx*k; G += g*k
            K += k

    if K != 0:
        return I/K, Dy/K, Dx/K, G/K


def comp_angle(_A, A):  # rn doesn't matter for angles

    _angle, angle = [atan2(Dy, Dx) for Dy, Dx in [_A, A]]
    dangle = _angle - angle  # difference between angles

    if dangle > pi: dangle -= 2*pi  # rotate full-circle clockwise
    elif dangle < -pi: dangle += 2*pi  # rotate full-circle counter-clockwise

    mangle = (cos(dangle)+1)/2  # angle similarity, scale to [0,1]
    dangle /= 2*pi  # scale to the range of mangle, signed: [-.5,.5]

    return [mangle, dangle]

def project(y, x, s, c, r):
    dist = s*y + c*x - r
    # Subtract left and right side by dist:
    # 0 = s*y + c*x - r - dist
    # 0 = s*y + c*x - r - dist*(s*s + c*c)
    # 0 = s*(y - dist*s) + c*(x - dist*c) - r
    # therefore, projection of y, x onto the line is:
    return y - dist*s, x - dist*c

if __name__ == "__main__":

    image_file = '../images/raccoon_eye.jpeg'
    image = imread(image_file)

    frame = CsliceEdge(image).segment()
    # verification:
    import matplotlib.pyplot as plt

    # show first largest n edges
    edge_, edgeQue = [], list(frame.blob_)
    while edgeQue:
        blob = edgeQue.pop(0)
        if hasattr(blob, "P_"): edge_ += [blob]
        elif hasattr(blob, "rlay"): edgeQue += blob.rlay.blob_

    num_to_show = 5
    show_gradient = True
    show_slices = True
    sorted_edge_ = sorted(edge_, key=lambda edge: len(edge.yx_), reverse=True)
    for edge in sorted_edge_[:num_to_show]:
        yx_ = np.array(edge.yx_)
        yx0 = yx_.min(axis=0) - 1

        # show edge-blob
        shape = yx_.max(axis=0) - yx0 + 2
        mask_nonzero = tuple(zip(*(yx_ - yx0)))
        mask = np.zeros(shape, bool)
        mask[mask_nonzero] = True
        plt.imshow(mask, cmap='gray', alpha=0.5)
        plt.title(f"area = {edge.area}")

        # show gradient
        if show_gradient:
            vu_ = [(-gy/g, gx/g) for i, gy, gx, g in edge.dert_.values()]
            y_, x_ = zip(*(yx_ - yx0))
            v_, u_ = zip(*vu_)
            plt.quiver(x_, y_, u_, v_, scale=100)

        # show slices
        if show_slices:
            for P in edge.P_:
                y_, x_ = zip(*(P.yx_ - yx0))
                if len(P.yx_) == 1:
                    v, u = P.axis
                    y_ = y_[0]-v/2, y_[0]+v/2
                    x_ = x_[0]-u/2, x_[0]+u/2
                plt.plot(x_, y_, "k-", linewidth=3)
                yp, xp = P.yx - yx0
                for link in P.link_[0]:
                    _yp, _xp = link.node_[0].yx - yx0
                    print(link.node_)
                    plt.plot([_xp, xp], [_yp, yp], "ko--", alpha=0.5)

                y_, x_ = zip(*([yx for yx in edge.rootd if edge.rootd[yx] is P] - yx0))
                plt.plot(x_, y_, 'o', alpha=0.5)


        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()