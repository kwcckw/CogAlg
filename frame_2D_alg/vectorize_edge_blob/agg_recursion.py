import numpy as np
from copy import deepcopy, copy
from itertools import combinations, zip_longest
from .slice_edge import comp_angle, CsliceEdge
from .comp_slice import comp_slice, comp_latuple, add_lat, CH, CG
from utils import extend_box
from frame_blobs import CBase

'''
Blob edges may be represented by higher-composition patterns, etc., if top param-layer match,
in combination with spliced lower-composition patterns, etc, if only lower param-layers match.
This may form closed edge patterns around flat core blobs, which defines stable objects. 

Graphs (predictive patterns) are formed from edges that match over < extendable max distance, 
then internal cross-comp rng/der is incremented per relative M/D: induction from prior cross-comp
(no lateral prediction skipping: it requires overhead that can only be justified in vertical feedback) 
- 
Primary value is match, diff.patterns borrow value from proximate match patterns, canceling their projected match. 
Thus graphs are assigned adjacent alt-fork graphs, to which they lend predictive value.
But alt match patterns borrow already borrowed value, which may be too tenuous to track, we use average borrowed value.
-
Clustering criterion is also M|D, summed across >ave vars if selective comp  (<ave vars are not compared, don't add costs).
Clustering is exclusive per fork,ave, with fork selected per var| derLay| aggLay 
Fuzzy clustering is centroid-based only, connectivity-based clusters will merge.
Param clustering if MM, along derivation sequence, or centroid-based if global?

There are concepts that include same matching vars: size, density, color, stability, etc, but in different combinations.
Weak value vars are combined into higher var, so derivation fork can be selected on different levels of param composition.
Clustering by variance: lend|borrow, contribution or counteraction to similarity | stability, such as metabolism? 
-
graph G:
Agg+ cross-comps top Gs and forms higher-order Gs, adding up-forking levels to their node graphs.
Sub+ re-compares nodes within Gs, adding intermediate Gs, down-forking levels to root Gs, and up-forking levels to node Gs.
-
Generic graph is a dual tree with common root: down-forking elements and up-forking clusters of this graph. 
This resembles a neuron, which has dendritic tree as input and axonal tree as output. 
But we have recursively structured param sets packed in each level of these trees, which don't exist in neurons.

Diagrams: 
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/agg_recursion_unfolded.drawio.png
'''
ave = 3
ave_d = 4
ave_L = 4
aves = [3,4]  # ave_Gm, ave_Gd
max_dist = 2

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"

    def __init__(l, nodet=None,derH=None, S=0, A=None, box=None, md_t=None, H_=None, root=None):
        super().__init__()
        # CL = binary tree of Gs, depth+/der+: CL nodet is 2 Gs, CL + CLs in nodet is 4 Gs, etc.,
        # unpack sequentially
        l.nodet = [] if nodet is None else nodet  # e_ in kernels, else replaces _node,node: not used in kernels
        l.A = [0,0] if A is None else A  # dy,dx between nodet centers
        l.S = 0 if S is None else S  # span: distance between nodet centers, summed into sparsity in CGs
        l.area = 0  # sum nodet
        l.box = [] if box is None else box  # sum nodet
        l.derH = CH(root=l) if derH is None else derH
        l.H_ = [] if H_ is None else H_  # if agg++| sub++?
        l.ft = [0,0]  # fork inclusion tuple, may replace Vt:
        l.Vt = [0,0]  # for rim-overlap modulated segmentation, init derH.Et[:2]
        l.n = 1  # min(node_.n)
        l.Et = [0,0,0,0]
        l.root = root
        # add rimt_, elay if der+
    def __bool__(l): return bool(l.derH.H)


def vectorize_root(image):  # vectorization in 3 composition levels of xcomp, cluster:

    frame = CsliceEdge(image).segment()
    for edge in frame.blob_:
        if (hasattr(edge, 'P_') and
            edge.latuple[-1] * (len(edge.P_)-1) > ave):
            comp_slice(edge)
            # init for agg+:
            edge.derH = CH(H=[CH()]); edge.derH.H[0].root = edge.derH; edge.fback_ = []; edge.Et = [0,0,0,0]
            # convert select PPs to Gs:
            if edge.mdLay.Et[0] * (len(edge.node_)-1)*(edge.rng+1) > ave * edge.mdLay.Et[2]:
                pruned_Q = []
                for PP in edge.node_:  # PP -> G
                    if PP.mdLay and PP.mdLay.Et[0] > ave * PP.mdLay.Et[2]:  # v>ave*r
                        PP.node_ = PP.P_  # revert node_t?
                        y0,x0,yn,xn = PP.box
                        PP.yx = [(y0+yn)/2, (x0+xn)/2]
                        PP.aRad = np.hypot(*np.subtract(PP.yx,(yn,xn)))
                        PP.Et = [0,0,0,0]  # [] in comp_slice
                        pruned_Q += [PP]
                if len(pruned_Q) > 10:
                    # discontinuous PP_ xcomp, cluster:
                    agg_recursion(edge, pruned_Q, fd=0)

def agg_recursion(root, Q, fd):  # breadth-first rng++ cross-comp -> eval cluster, fd recursion

    N__,L__,Et,rng = rng_link_(Q) if fd else rng_node_(Q)  # cross-comp PP_/ edge|frame, sub++/ L_
    m,d,mr,dr = Et
    fvd = d > ave_d * dr*(rng+1); fvm = m > ave * mr*(rng+1)
    if fvd or fvm:
        root.Et = np.add(root.Et,Et); L_ = [L for L_ in L__ for L in L_]
        # root.derH += L.derH:
        if fd: root.derH.append_(CH().append_(CH().copy(L_[0].derH)))  # new rngLay, aggLay
        else:  root.derH.H[-1].append_(L_[0].derH)  # append last aggLay
        for L in L_[1:]:
            root.derH.H[-1].H[-1].add_H(L.derH)  # accum Lay
        # rng_link_:
        if fvd and len(L_) > ave_L:  # comp L, sub-cluster by dL: mL is redundant to mN?
            set_attrs(L_,root)
            agg_recursion(root, L_,fd=1)  # appends last aggLay, L_=lG_ if segment
        # rng_node_:
        Q = N__[0]  # rng+ N_s are redundant, unless rngH xcomp, sub-cluster
        if fvm and len(Q) > ave_L:  # cluster ave_L != xcomp ave_L?
            Q[:] = seq_segment(root, Q, fd,rng)  # Q = nG_
            if len(Q) > ave_L:
                agg_recursion(root, Q,fd=0)  # adds higher aggLay / recursive call
'''
    if flat derH:
    root.derH.append_(CH().copy(L_[0].derH))  # init
    for L in L_[1:]: root.derH.H[-1].add_H(L.derH)  # accum
'''

def rng_node_(_N_):  # rng+ forms layer of rim_ and extH per N, appends N__,L__,Et, ~ graph CNN without backprop
                     # low-rng N_s are dense: sub-cluster or rng_link_-> segment/mS in G.rim_) graph| L_?
    N__ = []; L__ = []; ET = [0,0,0,0]
    rng = 1
    while True:
        N_ = []; Et = [0,0,0,0]
        # full search, no mediation:
        for _G,G in combinations(_N_,r=2):
            if _G in G.visited_:  # compared in any rng++
                continue
            dy,dx = np.subtract(_G.yx,G.yx)
            dist = np.hypot(dy,dx)
            aRad = (G.aRad+_G.aRad) / 2  # ave G radius
            # eval relative distance between G centers:
            if dist / max(aRad,1) <= (max_dist * rng):
                for _g, g in (_G,G),(G,_G):
                    g.visited_ += [_g]
                Link = CL(nodet=[_G,G], S=2, A=[dy,dx], box=extend_box(G.box,_G.box))
                comp_N(Link, Et, rng)
                if Link.Et[0] > ave * Link.Et[2] * (rng+1):
                    for g in _G,G:
                        if g not in N_: N_ += [g]
        if Et[0] > ave * Et[2] * rng:
            ET = np.add(ET, Et)
            L__ += [list(set([Lt[0] for N in N_ for Lt in N.rim_[-1]]))]
            N__ += [N_]  # nest to sub-cluster?
            _N_ = N_
            rng += 1
        else:
            break
    return N__,L__,ET,rng

def rng_link_(iL_):  # comp CLs: der+'rng+ in root.link_ rim_t node rims: directional and node-mediated link tracing

    _N_t_ = [[[L.nodet[0]],[L.nodet[1]]] for L in iL_]  # Ns are rim-mediating nodes, starting with L.nodet
    ET = [0,0,0,0]; L__ = []; LL__ = []  # all links between Ls in potentially extended L__
    rng = 1; _L_ = iL_[:]
    while True:
        L_,LL_,Et = [],[],[0,0,0,0]
        N_t_ = [[[],[]] for _ in _L_]  # new rng lay of mediating nodes, traced from all prior layers?
        for L, _N_t, N_t in zip(_L_, _N_t_, N_t_):
            for rev, _N_, N_ in zip((0,1), _N_t, N_t):
                # comp L, _L incrementally mediated by nodets, flatten rim_s:
                rim_ = [rim for n in _N_ for rim in (n.rim_ if isinstance(n, CG) else [n.rimt_[0][0] + n.rimt_[0][1]])]
                for rim in rim_:
                    for _L,_rev in rim:  # _L is reversed relative to its 2nd node
                        if _L is L or _L in L.visited_: continue
                        if _L not in iL_: set_attrs([_L],_L_[0].root)
                        L.visited_ += [_L]; _L.visited_ += [L]
                        Link = CL(nodet=[_L,L], S=2, A=np.subtract(_L.yx,L.yx), box=extend_box(_L.box, L.box))
                        comp_N(Link, Et, rng, rev^_rev)  # L.rim_t +=new Link, d = -d if one L is reversed
                        if Link.Et[0] > ave * Link.Et[2] * (rng+1):
                            # L += rng+ -mediating nodes, link orders: nodet < L < rimt_, mN.rim || L
                            N_ += _L.nodet  # get _Ls in N_ rims
                            if _L not in _L_:
                                _L_ += [_L]; N_t_ += [[[],[]]]  # not in root
                            N_t_[_L_.index(_L)][1-rev] += L.nodet
                            L_ += [_L]; LL_ += [Link]
        if L_:
            ET = np.add(ET,Et); L__ += [L_]; LL__ += [LL_]
            V = 0; L_,_N_t_ = [],[]
            for L, N_t in zip(_L_,N_t_):
                if any(N_t):
                    L_ += [L]; _N_t_ += [N_t]
                    V += L.derH.Et[0] - ave * L.derH.Et[2] * rng
            if V > 0:  # rng+ if vM of extended N_t_
                _L_ = L_; rng += 1
            else:
                break
    return L__, LL__, ET, rng

def comp_N(Link, iEt, rng, rev=None):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = rev is not None  # compared links have binary relative direction?
    _N, N = Link.nodet
    _L,L = (2,2) if fd else (len(_N.node_),len(N.node_)); _S,S = _N.S,N.S; _A,A = _N.A,N.A
    if rev: A = [-d for d in A]  # reverse angle direction if N is left link?
    rn = _N.n / N.n
    mdext = comp_ext(_L,L, _S,S/rn, _A,A); md_t = [mdext]; Et = mdext.Et; Rt = mdext.Rt; n = mdext.n
    if not fd:  # CG
        mdlat = comp_latuple(_N.latuple, N.latuple, rn, fagg=1); md_t += [mdlat]; Et = np.add(Et,mdlat.Et); Rt = np.add(Rt,mdlat.Rt); n += mdlat.n
        mdLay = _N.mdLay.comp_md_(N.mdLay, rn, fagg=1);          md_t += [mdLay]; Et = np.add(Et,mdLay.Et); Rt = np.add(Rt,mdLay.Rt); n += mdLay.n
    # | n = (_n+n) /2?:
    elay = CH( H=[CH(n=n, md_t=md_t, Et=Et, Rt=Rt)], n=n, md_t=[CH().copy(md_) for md_ in md_t], Et=copy(Et),Rt=copy(Rt))
    if _N.derH and N.derH:
        dderH = _N.derH.comp_H(N.derH, rn, fagg=1)  # comp shared layers
        elay.append_(dderH, flat=1)
    Et = elay.Et
    iEt[:] = np.add(iEt,Et); N.Et[:] = np.add(N.Et,Et); _N.Et[:] = np.add(_N.Et,Et)
    Link.derH = elay; elay.root = Link; Link.Et = Et; Link.n = min(_N.n,N.n)
    Link.nodet = [_N,N]; Link.yx = np.add(_N.yx,N.yx) /2
    # S,A set before
    for rev, node in zip((0,1),(_N,N)):  # ?reversed Link direction

        # fadd == 1
        if (len(node.rimt_) if fd else len(node.rim_)) == rng:  # we can check with length rim instead of rng
            node.extH.H[-1].add_H(elay)
            if fd: node.rimt_[-1][1-rev] += [[Link,rev]]  # add in last rng layer, opposite to _N,N dir
            else:  node.rim_[-1] += [[Link, rev]]
        # fadd == 0
        else:
            node.extH.append_(elay)
            if fd: node.rimt_ = [[[[Link,rev]],[]]] if dir else [[[],[[Link,rev]]]]  # add rng layer
            else:  node.rim_ += [[[Link, rev]]]

def comp_ext(_L,L,_S,S,_A,A):  # compare non-derivatives:

    dL = _L - L; mL = min(_L,L) - ave_L  # direct match
    dS = _S/_L - S/L; mS = min(_S,S) - ave_L  # sparsity is accumulated over L
    mA, dA = comp_angle(_A,A)  # angle is not normalized
    M = mL + mS + mA
    D = abs(dL) + abs(dS) + abs(dA)  # signed dA?
    mrdn = M > D; drdn = D<= M
    mdec = mL/ max(L,_L) + mS/ max(S,_S) if S or _S else 1 + mA  # Amax = 1
    ddec = max_dist + mL/ (L+_L) + dS/ (S+_S) if S or _S else 1 + dA

    return CH(H=[mL,dL, mS,dS, mA,dA], Et=[M,D,mrdn,drdn], Rt=[mdec,ddec], n=0.5)


def seq_segment(root, Q, fd, rng):  # cluster Q:G_|L_, by density of shared links

    N_, max_ = [], []
    for N in Q:
        lrim = [Lt[0] for Lt in (N.rimt_[-1][0] + N.rimt_[-1][1] if fd else N.rim_[-1])]  # external links
        N.lrim = [L for L in lrim if L.Et[fd] > ave * (L.Et[2+fd]) * rng]  # +ve only
        N.nrim = [_N for L in N.lrim for _N in L.nodet if _N is not N]  # external nodes
        N_ += [N]
        # select exemplar maxes to segment clustering:
        emax_ = [eN for eN in N.nrim if eN.Et[fd] >= N.Et[fd] or eN in max_]  # _N if _N == N
        # if max doesn't have lrim, it will not be merged with other max Gt, so max should have at least 1 lrim?
        if not emax_ and N.lrim:  # no higher-val neighbors, extended rrim max: V * k * max_rng?
            max_ += [N]
    Gt_ = []
    for N in (max_ if max_ else N_):  # for those not mediated non max N_, they will be just ignored?
        _nrim_ = N.nrim; _lrim_ = N.lrim
        node_ = {N}; link_ = set(); Et = [0,0,0,0]
        while _nrim_:
            nrim_,lrim_ = set(),set()
            for _N, _L in zip(_nrim_, _lrim_):  # recursive merge connected Ns
                if _N not in N_: continue  # was merged
                # density-based clustering criterion:
                if len(N.nrim) > ave_L or len(_N.nrim) > ave_L and _L in _N.lrim:  # we can include connectivity too? By checking _L in _N.lrim?
                    node_.add(_N); link_.add(_L); Et = np.add(Et, _L.Et)
                    nrim_.update(set(_N.nrim) - node_)
                    lrim_.update(set(_N.lrim) - link_)
                    N_.remove(_N)
            # for next loop:
            _nrim_, _lrim_ = nrim_, lrim_
        Gt_ += [[list(node_), list(link_), Et]]  # convert to list

    return [sum2graph(root, Gt, fd, rng) for Gt in Gt_]


def segment(root, Q, fd, rng):  # cluster iN_(G_|L_) by density of shared links

    N_, max_ = [],[]
    # init Gt per G|L node:
    for N in Q:
        Lrim = [Lt[0] for Lt in (N.rimt_[-1][0] + N.rimt_[-1][1] if fd else N.rim_[-1])]  # external links
        Lrim = [L for L in Lrim if L.Et[fd] > ave * (L.Et[2+fd]) * rng]  # +ve to merge Gts
        Nrim = [_N for L in Lrim for _N in L.nodet if _N is not N]  # external nodes
        Gt = [[N],[], Lrim, Nrim, [0,0,0,0]]
        N.root = Gt
        N_ += [Gt]
        # select exemplar maxes to segment clustering:
        emax_ = [eN for eN in Nrim if eN.Et[fd] >= N.Et[fd] or eN in max_]  # _N if _N == N
        if not emax_: max_ += [Gt]  # N.root, if no higher-val neighbors
        # extended rrim max: V * k * max_rng?
    for Gt in N_: Gt[3] = [_N.root for _N in Gt[3]]  # replace eNs with Gts
    for Gt in max_ if max_ else N_:
        node_,link_, Lrim, Nrim, Et = Gt
        while True:  # while Nrim
            _Nrim_,_Lrim_ = [],[]  # recursive merge connected Gts
            for _Gt,_L in zip(Nrim,Lrim):  # always single N unless parallelized clustering
                if _Gt not in N_: continue  # was merged
                for L in _Gt[2]:
                    if L in Lrim and (len(_Lrim_) > ave_L or len(Lrim) > ave_L):  # density-based
                        merge(Gt,_Gt, _Nrim_,_Lrim_); N_.remove(_Gt)
                        break  # merge if any +ve shared external links
            if _Nrim_:
                Nrim[:],Lrim[:] = _Nrim_,_Lrim_  # for clustering, else break, contour = term rims?
            else: break
    return [sum2graph(root, Gt, fd, rng) for Gt in N_]

def merge(Gt,gt, _Nrim_,_Lrim_):

    N_,L_, Lrim, Nrim, Et = Gt
    n_,l_, lrim, nrim, et = gt
    for N in N_:
        N.root = Gt
    Et[:] = np.add(Et, et)
    N_ += n_ # internal, no overlap
    L_ += l_
    for N, L in zip(nrim, lrim):
        if N not in Nrim and N is not Gt:
            _Nrim_ += [N]; _Lrim_ += [L]  # aligned
            # add shared L to L_?

def set_attrs(Q, root):

    for e in Q:
        e.visited_ = []
        if isinstance(e, CL):
            e.rimt_ = []  # nodet-mediated links, same der order as e
            e.root = root
        if hasattr(e,'extH'): e.derH.append_(e.extH)  # no default CL.extH
        else: e.extH = CH()  # set in sum2graph
        e.Et = [0,0,0,0]
        e.aRad = 0
    return Q

def sum2graph(root, grapht, fd, rng):  # sum node and link params into graph, aggH in agg+ or player in sub+

    N_, L_, Et = grapht  # [node_, link_, Et]
    # flattened N__, L__ if segment / rng++
    graph = CG(fd=fd, root = root, node_=N_, link_=L_, rng=rng, Et=Et)
    yx = [0,0]
    lay0 = CH(node_= N_)  # comparands, vs. L_: summands?
    for link in L_:  # unique current-layer mediators: Ns if fd else Ls
        graph.S += link.S
        graph.A = np.add(graph.A,link.A)  # np.add(graph.A, [-link.angle[0],-link.angle[1]] if rev else link.angle)
        lay0.add_H(link.derH) if lay0 else lay0.append_(link.derH)
    graph.derH.append_(lay0)  # empty for single-node graph
    derH = CH()
    for N in N_:
        graph.n += N.n  # +derH.n
        graph.area += N.area
        graph.box = extend_box(graph.box, N.box)
        yx = np.add(yx, N.yx)
        if N.derH: derH.add_H(N.derH)
        if isinstance(N,CG):
            graph.mdLay.add_md_(N.mdLay)
            add_lat(graph.latuple, N.latuple)
        N.root = graph
    graph.derH.append_(derH, flat=1)  # comp(derH) forms new layer, higher layers are added by feedback
    L = len(N_)
    yx = np.divide(yx,L); graph.yx = yx
    # ave distance from graph center to node centers:
    graph.aRad = sum([np.hypot(*np.subtract(yx,N.yx)) for N in N_]) / L
    if fd:
        # assign alt graphs from d graph, after both linked m and d graphs are formed
        for node in graph.node_:  # CG or CL
            mgraph = node.root
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]
    return graph