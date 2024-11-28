import sys
sys.path.append("..")
import numpy as np
from copy import copy, deepcopy
from itertools import combinations
from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_
from comp_slice import comp_latuple, comp_md_
from trace_edge import comp_N, comp_node_, comp_link_, sum2graph, get_rim, CH, CG, ave, ave_d, ave_L, vectorize_root, comp_area, extend_box, ave_rn
'''
Cross-compare and cluster edge blobs within a frame,
potentially unpacking their node_s first,
with recursive agglomeration
'''

def agg_cluster_(frame):  # breadth-first (node_,L_) cross-comp, clustering, recursion

    def cluster_eval(frame, N_, fd):

        pL_ = {l for n in N_ for l,_ in get_rim(n, fd)}
        if len(pL_) > ave_L:
            G_ = cluster_N_(frame, pL_, fd)  # optionally divisive clustering
            frame.subG_ = G_
            if len(G_) > ave_L:
                get_exemplar_(frame)  # may call centroid clustering
    '''
    cross-comp converted edges, then GGs, GGGs, etc, interlaced with exemplar selection 
    '''
    N_,L_,(m,d,r) = comp_node_(frame.subG_)  # exemplars, extrapolate to their Rims?
    if m > ave * r:
        mlay = CH().add_H([L.derH for L in L_])  # mfork, else no new layer
        frame.derH = CH(H=[mlay], md_t=deepcopy(mlay.md_t), Et=copy(mlay.Et), n=mlay.n, root=frame); mlay.root=frame.derH  # init
        vd = d - ave_d * r
        if vd > 0:
            for L in L_:
                L.root_ = [frame]; L.extH = CH(); L.rimt = [[],[]]
            lN_,lL_,md = comp_link_(L_)  # comp new L_, root.link_ was compared in root-forming for alt clustering
            vd *= md / ave
            frame.derH.append_(CH().add_H([L.derH for L in lL_]))  # dfork
            # recursive der+ eval_: cost > ave_match, add by feedback if < _match?
        else:
            frame.derH.H += [[]]  # empty to decode rng+|der+, n forks per layer = 2^depth
        # + aggLays and derLays, each has exemplar selection:
        cluster_eval(frame, N_, fd=0)
        if vd > 0: cluster_eval(frame, lN_, fd=1)


def cluster_N_(root, L_, fd, nest=1):  # top-down segment L_ by >ave ratio of L.dists

    ave_rL = 1.2  # defines segment and cluster

    L_ = sorted(L_, key=lambda x: x.dist, reverse=True)  # lower-dist links
    _L = L_[0]
    N_, et = {*_L.nodet}, _L.derH.Et
    # current dist segment:
    for i, L in enumerate(L_[1:], start=1):  # long links first
        rel_dist = _L.dist / L.dist  # >1
        if rel_dist < ave_rL or et[0] < ave or len(L_[i:]) < ave_L:  # ~=dist Ns or either side of L is weak
            _L = L; et += L.derH.Et
            for n in L.nodet: N_.add(n)  # in current dist span
        else: break  # terminate contiguous-distance segment
    min_dist = _L.dist
    Gt_ = []
    for N in N_:  # cluster current distance segment
        if len(N.root_) > nest: continue  # merged, root_[0] = edge
        node_,link_, et = set(),set(), np.array([.0,.0, 1.0])
        Gt = [node_,link_,et, min_dist]; N.root_ += [Gt]
        _eN_ = {N}
        while _eN_:
            eN_ = set()
            for eN in _eN_:  # cluster rim-connected ext Ns, all in root Gt
                node_.add(eN)  # of all rim
                eN.root_ += [Gt]
                for L,_ in get_rim(eN, fd):
                    if L not in link_:
                        # if L.derH.Et[0]/ave * n.extH m/ave or L.derH.Et[0] + n.extH m*.1: density?
                        eN_.update([n for n in L.nodet if len(n.root_) <= nest])
                        if L.dist >= min_dist:
                            link_.add(L); et += L.derH.Et
            _eN_ = eN_
        sub_L_ = set()  # form subG_ from shorter-L seg per Gt, depth-first:
        for N in node_:  # cluster in N_
            sub_L_.update({l for l,_ in get_rim(N,fd) if l.dist < min_dist})
        if len(sub_L_) > ave_L:
            subG_ = cluster_N_(Gt, sub_L_, fd, nest+1)
        else: subG_ = []
        Gt += [subG_]; Gt_ += [Gt]
    G_ = []
    for Gt in Gt_:
        node_, link_, et, minL, subG_ = Gt; Gt[0] = list(node_)
        if et[0] > et[2] *ave *nest:  # rdn incr/ dist decr
            G_ += [sum2graph(root, Gt, fd, nest)]
        else:
            for n in node_: n.root_.pop()
    return G_

'''
 Connectivity clustering terminates at effective contours: alt_Gs, beyond which cross-similarity is not likely to continue. 
 Next cross-comp is discontinuous and should be selective for well-defined clusters: stable and likely recurrent.
 
 Such exemplar selection is by global similarity in centroid clustering, vs. transitive similarity in connectivity clustering.
 It's a compressive learning phase, while connectivity clustering is generative, forming new derivatives and composition levels.
 
 Centroid clusters may be extended, but only their exemplars will be cross-compared on the next connectivity clustering level.
 Other nodes in the cluster can be predicted from the exemplar, they are the same type of objects. 
'''
def get_exemplar_(frame):

    def comp_cN(_N, N):  # compute match without new derivatives: global cross-comp is not directional

        rn = _N.n / N.n
        mL = min(len(_N.node_),len(N.node_)) - ave_L
        mA = comp_area(_N.box, N.box)[0]
        mLat = comp_latuple(_N.latuple, N.latuple, rn,fagg=1)[1][0]
        mLay = comp_md_(_N.mdLay[0], N.mdLay[0], rn)[1][0]
        HEt = _N.derH.comp_H(N.derH, rn).Et if _N.derH and N.derH else np.zeros(3)
        # comp node_, comp altG from converted adjacent flat blobs?

        return mL+mA+mLat+mLay+HEt[0], HEt[2]

    def xcomp_(N_):  # initial global cross-comp
        for g in N_: g.M, g.Mr = 0,0  # setattr

        for _G, G in combinations(N_, r=2):
            rn = _G.n/G.n
            if rn > ave_rn: continue  # scope disparity
            # use regular comp_N and compute Links, we don't need to compare again in eval_overlap?
            dy,dx = np.subtract(_G.yx,G.yx)  # no dist eval
            Link = comp_N(_G,G, _G.n/G.n, angle=[dy,dx], dist=np.hypot(dy,dx))
            # non selective, will be prune in centrtoid_cluster later
            for N in (_G, G):
                N.Rim.add(Link)
                N.M += Link.derH.Et[0]
 
    def centroid(node_):  # sum and average Rim nodes

        C = CG()
        for n in node_:
            C.n += n.n; C.Et += n.Et; C.rng = n.rng; C.aRad += n.aRad; C.box = extend_box(C.box,n.box)
            C.latuple += n.latuple; C.mdLay += n.mdLay; C.derH.add_H(n.derH); C.extH.add_H(n.extH)
        # get averages:
        k = len(node_); C.n/=k; C.Et/=k; C.latuple/=k; C.mdLay/=k; C.aRad/=k; C.derH.norm_(k) # derH/=k
        return C

    def centroid_cluster(N):  # refine Rim to convergence
        _Rim, _peRim,_M = N.Rim, N.Rim, N.M  # no need for Mr here? I think we need it in the dM eval below?

        dM = ave + 1
        while dM > ave * max(1, N.Mr):
            Rim, peRim, M = set(), set(),  0
            C = centroid({n for L in _Rim for n in L.nodet})
            for L in _peRim:
                _N, m = L.nodet[0] if L.nodet[1] is N else L.nodet[1], L.derH.Et[0]
                mm,rr = comp_cN(C,_N)
                if mm > ave * rr:
                    peRim.add(L)
                    if mm > ave * rr * 10:  # copy ref from perim to Rim
                        Rim.add(L); M += m
            dM = M - _M
            _Rim,_peRim,_M = Rim, peRim, M

        N.Rim, _peRim, N.M = list(Rim),list(peRim), M

    def eval_overlap(N):  # check for reciprocal refs in _Ns, compare to diffs, remove the weaker ref if <ave diff

        fadd = 1
        for L in copy(N.Rim):
            _N, _m = L.nodet[0] if L.nodet[1] is N else L.nodet[1], L.derH.Et[0]
            if _N in N.compared_: continue  # also skip in next agg+
            N.compared_ += [_N]; _N.compared_ += [N]
            for _L in copy(_N.Rim):
                __N, __m = _L.nodet[0] if _L.nodet[1] is _N else _L.nodet[1], _L.derH.Et[0]
                if __N is N:  # reciprocal to ref
                    dy,dx = np.subtract(_N.yx,N.yx)  # no dist eval
                    # replace Rim refs with links in xcomp_, instead of comp_N here?
                    Link = comp_N(_N,N, _N.n/N.n, angle=[dy,dx], dist=np.hypot(dy,dx))
                    minN, minL = (_N,_L) if N.M > _N.M else (N,L)
                    if Link.derH.Et[1] < ave_d:
                        # exemplars are similar, remove min
                        minN.Rim.remove(minL); minN.M -= Link.derH.Et[0]
                        if N is minN: fadd = 0
                    else:  # exemplars are different, keep both
                        if N is minN: N.Mr += 1
                        _N.extH.add_H(Link.derH), N.extH.add_H(Link.derH)
                    break
        return fadd

    def prune_overlap(N_):  # select Ns with M > ave * Mr

        for N in N_:  # connectivity cluster N.Rim may be represented by multiple sufficiently different exemplars
            if N.M > ave * 10:  # or if len(N.perim) / len(N.Rim) > ave_L?
                centroid_cluster(N)  # refine N.Rim
        exemplar_ = []
        for N in N_:
            if eval_overlap(N) and N.M + N.Et[0] > ave * (N.Et[2] + N.Mr/len(N.Rim)):  # normalize Mr
                exemplar_ += [N]
        return exemplar_

    N_ = frame.subG_  # complemented Gs: m-type core + d-type contour
    for N in N_:
        N.perim = set(); N.Rim = set(); N.root_ += [frame]; N.M = 0; N.Mr = 0; N.compared_ = []
    xcomp_(N_)
    frame.subG_ = prune_overlap(N_)  # select strong Ns as exemplars of their Rim
    if len(frame.subG_) > ave_L:
        agg_cluster_(frame)  # selective connectivity clustering between exemplars, extrapolated to Rim?


if __name__ == "__main__":
    image_file = '../images/raccoon_eye.jpeg'
    image = imread(image_file)
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vectorize_root(frame)
    if frame.subG_:  # converted edges
        get_exemplar_(frame)  # selects connectivity-clustered edges for agg_cluster_