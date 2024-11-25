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
    cross-comp converted edges, then GGs, GGGs, etc, interlaced with exemplar selection: 
    '''
    N_,L_,(m,d,mr,dr) = comp_node_(frame.subG_)  # exemplars, extrapolate to their Rims?
    if m > ave * mr:
        mlay = CH().add_H([L.derH for L in L_])  # mfork, else no new layer
        frame.derH = CH(H=[mlay], md_t=deepcopy(mlay.md_t), Et=copy(mlay.Et), n=mlay.n, root=frame); mlay.root=frame.derH  # init
        vd = d * (m/ave) - ave_d * dr  # vd = proportional borrow from co-projected m
        # proj_m = m - vd, no generic rdn~ave_d?
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
        node_,link_, et = set(),set(), np.zeros(4)
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
        mderH = _N.derH.comp_H(N.derH, rn).Et[0] if _N.derH and N.derH else 0
        # comp node_?
        # comp alt_graph_, from converted adjacent flat blobs?
        return mL + mA + mLat + mLay + mderH

    def xcomp_(N_):  # initial global cross-comp
        for g in N_: g.M = 0  # setattr

        for _G, G in combinations(N_, r=2):
            rn = _G.n/G.n
            if rn > ave_rn: continue  # scope disparity
            M = comp_cN(_G, G)
            vM = M - ave
            for _g,g in (_G,G),(G,_G):
                if vM > 0:
                    g.perim.add((_g,M))  # loose match ref: unilateral link
                    if vM > ave:
                        g.Rim.add((_g,M))  # strict match ref
                        g.M += M

    def centroid(node_):  # sum and average Rim nodes

        C = CG()
        for n,_ in node_:
            C.n += n.n; C.rng = n.rng; C.aRad += n.aRad; C.box = extend_box(N.box, n.box)
            C.latuple += n.latuple; C.mdLay += n.mdLay; C.derH.add_H(n.derH); C.extH.add_H(n.extH)
        # get averages:
        k = len(node_); C.n /= k; C.latuple /= k; C.mdLay /= k; C.aRad /= k; C.derH.norm_(k)  # derH/= k
        return C

    def eval_overlap(N):

        for ref in N.Rim:
            _N, _m = ref
            for _ref in copy(_N.Rim):
                if _ref[0] is N:  # reciprocal to ref
                    dy, dx = np.subtract(_N.yx, N.yx); dist = np.hypot(dy, dx)
                    Link = CL(nodet=[_N,N], angle=[dy,dx], dist=dist, box=extend_box(N.box,_N.box))
                    m,d,mr,dr = comp_N(Link, _N.n/N.n)
                    if d < ave_d * dr:  # probably wrong eval
                        # exemplars are similar, remove min
                        minN,r,v = (_N,_ref,_m) if N.M > _N.M else (N,ref,m)
                        minN.Rim.remove(r); minN.M -= v
                    else:  # exemplars are different, keep both
                        _N.extH.add_H(Link), N.extH.add_H(Link)
                    break

    def prune_overlap(N_):  # select Ns with M > ave * Mr
        exemplar_ = []

        for N in N_:
            eval_overlap(N)
            if N.M > ave:
                if N.M > ave*10:
                    centroid_cluster(N)  # refine N.Rim
                exemplar_ += [N]

        return exemplar_

    # not updated with eval overlap:
    def centroid_cluster(N):
        _Rim,_perim,_M = N.Rim, N.perim, N.M

        dM = ave + 1  # refine Rim to convergence:
        while dM > ave:
            Rim, perim, M = set(), set(), 0
            C = centroid(_Rim)
            for ref in _perim:
                _N,m = ref
                mm = comp_cN(C,_N)
                if mm > ave:
                    perim.add((_N,m))
                    if mm > ave * 20:  # copy ref from perim to Rim, if not in stronger _N.Rim
                        fRim = 1
                        for _ref in copy(_N.Rim):
                            if _ref[0] is N:  # reciprocal to ref
                                fRim = 0
                                if N.M > _N.M:  # move ref from _N.Rim to N.Rim
                                    _N.Rim.remove(_ref); _N.M -= m
                                    fRim = 1; break
                        if fRim:  # not in stronger _N.Rim: strict link is represented in only one of its Ns
                            Rim.add(ref); M += m
            dM = M - _M
            _node_,_peri_,_M = Rim, perim, M

        N.Rim, N.perim, N.M = list(Rim),list(perim), M  # final cluster

    N_ = frame.subG_  # should be complemented graphs: m-type core + d-type contour
    for N in N_:
        N.perim = set(); N.Rim = set(); N.root_ += [frame]; N.M = 0
    xcomp_(N_)
    frame.subG_ = prune_overlap(N_)  # select strong Ns as exemplars of their Rim
    if len(frame.subG_) > ave_L:
        agg_cluster_(frame)  # selective connectivity clustering between exemplars


if __name__ == "__main__":
    image_file = '../images/raccoon_eye.jpeg'
    image = imread(image_file)
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vectorize_root(frame)
    if frame.subG_:  # converted edges
        get_exemplar_(frame)  # selects connectivity-clustered edges for agg_cluster_