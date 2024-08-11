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
Graphs are formed from blobs that match over < max distance. 
-
They are also assigned adjacent alt-fork graphs, which borrow predictive value from the graph. 
Primary value is match, diff.patterns borrow value from proximate match patterns, canceling their projected match. 
But alt match patterns borrow already borrowed value, which may be too tenuous to track, we use average borrowed value.
-
Clustering criterion is M|D, summed across >ave vars if selective comp (<ave vars are not compared and don't add costs).
Fork selection should be per var| der layer| agg level. Clustering is exclusive per fork,ave, overlapping between fork,aves.  
(same fork,ave fuzzy clustering is possible if centroid-based, connectivity-based clusters merge)

There are concepts that include same matching vars: size, density, color, stability, etc, but in different combinations.
Weak value vars are combined into higher var, so derivation fork can be selected on different levels of param composition.
Clustering by variance: lend|borrow, contribution or counteraction to similarity | stability, such as metabolism? 
-
graph G:
Agg+ cross-comps top Gs and forms higher-order Gs, adding up-forking levels to their node graphs.
Sub+ re-compares nodes within Gs, adding intermediate Gs, down-forking levels to root Gs, and up-forking levels to node Gs.
-
Generic graph is a dual tree with common root: down-forking input node Gs and up-forking output graph Gs. 
This resembles a neuron, which has dendritic tree as input and axonal tree as output. 
But we have recursively structured param sets packed in each level of these trees, which don't exist in neurons.

Diagrams: 
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/agg_recursion_unfolded.drawio.png
'''
ave = 3
ave_L = 4
G_aves = [4,6]  # ave_Gm, ave_Gd
max_dist = 2

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"

    def __init__(l, nodet=None,derH=None, span=0, angle=None, box=None, md_t=None, H_=None):
        super().__init__()
        # CL = binary tree of Gs, depth+/der+: CL nodet is 2 Gs, CL + CLs in nodet is 4 Gs, etc.,
        # unpack sequentially
        l.nodet = [] if nodet is None else nodet  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.angle = [0,0] if angle is None else angle  # dy,dx between nodet centers
        l.span = span  # distance between nodet centers
        l.area = 0  # sum nodet
        l.box = [] if box is None else box  # sum nodet
        l.latuple = []  # sum nodet
        l.md_t = [] if md_t is None else md_t  # [mdlat,mdLay,mdext] per layer
        l.derH = CH(root=l) if derH is None else derH
        l.H_ = [] if H_ is None else H_  # if agg++| sub++
        l.mdext = []  # Et, Rt, Md_
        l.ft = [0,0]  # fork inclusion tuple, may replace Vt:
        l.Vt = [0,0]  # for rim-overlap modulated segmentation, init derH.Et[:2]
        l.n = 1  # min(node_.n)
        l.S = 0  # sum nodet
        l.Et = [0,0,0,0]
        l.root = None
        # rimt_ = [[],[]]  # if der+
    def __bool__(l): return bool(l.derH.H)


def vectorize_root(image):  # vectorization in 3 composition levels of xcomp, cluster:

    frame = CsliceEdge(image).segment()
    for edge in frame.blob_:
        if hasattr(edge, 'P_') and edge.latuple[-1] * (len(edge.P_)-1) > G_aves[0]:  # eval PP, rdn=1  (ave_PPm or G_aves[0]?)
            comp_slice(edge)
            # init for agg+:
            edge.derH = CH(H=[CH()]); edge.derH.H[0].root = edge.derH
            edge.link_ = []; edge.fback_t = [[],[]]; edge.Et = [0,0,0,0]
            node_t, link_t = [[],[]], [[],[]]
            for fd, node_ in enumerate(copy(edge.node_)):  # always node_t
                if edge.mdLay.Et[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.mdLay.Et[2+fd]:
                    pruned_node_ = []
                    for PP in node_:  # PP -> G
                        if PP.mdLay and PP.mdLay.Et[fd] > G_aves[fd] * PP.mdLay.Et[2+fd]:  # v>ave*r
                            PP.root_ = []  # no feedback to edge?
                            PP.node_ = PP.P_  # revert node_
                            y0,x0,yn,xn = PP.box
                            PP.yx = [(y0+yn)/2, (x0+xn)/2]
                            PP.aRad = np.hypot(*np.subtract(PP.yx,(yn,xn)))
                            PP.Et = [0,0,0,0]  # [] in comp_slice
                            pruned_node_ += [PP]
                    if len(pruned_node_) > 10:  # discontinuous PP rng+ cross-comp, cluster -> G_t:
                        agg_recursion(edge, N_=pruned_node_, fL=0)
                        node_t[fd] = edge.node_  # edge.node_ is current fork node_t
                        link_t[fd] = edge.link_
            if any(node_t):
                edge.node_ = node_t; edge.link_ = link_t  # edge.node_ may be node_tt: node_t per fork?


def agg_recursion(root, N_, fL, rng=1):  # fL: compare node-mediated links, else < rng-distant CGs
    '''
    der+ forms Link_, link.node_ -> dual trees of matching links in der+rng+, complex higher links
    rng+ keeps N_ + distant M / G.M, less likely: term by >d/<m, less complex
    '''
    N_,L_,Et,rng = rng_link_(N_) if fL else rng_node_(N_, rng)  # N_ = L_ if fL else G_, both rng++
    if len(N_) > ave_L:
        node_t = form_graph_t(root, N_,L_, Et,rng)  # fork eval, depth-first sub+, sub_Gs += fback_
        if node_t:
            for fd, node_ in zip((0,1), node_t):  # after sub++ in form_graph_t
                N_ = []
                for N in node_:  # prune, select max node_ derH.i
                    derH = N.derH
                    if derH.Et[fd] > G_aves[fd] * derH.Et[2+fd]:
                        for i,lay in enumerate(sorted(derH.H, key=lambda lay:lay.Et[fd], reverse=True)):
                            di = lay.i-i  # lay.i: index in H, di: der rdn - value rdn
                            lay.Et[2+fd] += di  # update der rdn to val rdn
                            if not i:  # max val lay
                                N.node_ = lay.node_; derH.ii = lay.i  # exemplar lay index
                        N_ += [N]
                if root.derH.Et[0] * (max(0,(len(N_)-1)*root.rng)) > G_aves[1]*root.derH.Et[2]:
                    # agg+rng+, val *= n comparands, forms CGs:
                    agg_recursion(root, N_, fL=0)
            root.node_[:] = node_t
        # else keep root.node_

def rng_node_(_N_, rng):  # forms discrete rng+ links, vs indirect rng+ in rng_kern_, still no sub_Gs / rng+

    rN_ = []; rEt = [0,0,0,0]; n = 0

    while True:
        N_, Et = rng_kern_(_N_, rng)  # += rng layer
        if Et[0] > ave * Et[2]:
            _N_ = N_
            rEt = [V+v for V,v in zip(rEt, Et)]
            if not n: rN_ = N_  # 1st pruned N_
            rng += 1
            n += 1
        else:
            break
    rL_ = list(set([Lt[0]for N in rN_ for rim in N.rim_ for Lt in rim]))

    return rN_, rL_, rEt, rng

def rng_kern_(N_, rng):  # comp Gs summed in kernels, ~ graph CNN without backprop, not for CLs

    _G_ = []
    Et = [0,0,0,0]
    # comp_N:
    for (_G, G) in list(combinations(N_,r=2)):
        if _G in [G for visited_ in G.visited__ for G in visited_]:  # compared in any rng++
            continue
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        aRad = (G.aRad+_G.aRad) / 2  # ave G radius
        # eval relative distance between G centers:
        if dist / max(aRad,1) <= max_dist * rng:
            for _g,g in (_G,G),(G,_G):
                if len(g.extH.H)==rng: g.visited__[-1] += [_g]
                else: g.visited__ += [[_g]]  # init layer
            Link = CL(nodet=[_G,G], span=2, angle=[dy,dx], box=extend_box(G.box,_G.box))
            if comp_N(Link, Et, rng):
                for g in _G,G:
                    if g not in _G_: _G_ += [g]
    G_ = []  # init conv kernels:
    for G in (_G_):
        krim = []
        for link,rev in G.rim_[-1]:
            if link.ft[0]:  # must be mlink
                _G = link.nodet[0] if link.nodet[1] is G else link.nodet[1]
                krim += [_G]
                if hasattr(G,'dLay'): G.dLay.add_H(link.derH)
                else:                 G.dLay = CH().copy(link.derH)
                G._kLay = sum_kLay(G,_G); _G._kLay = sum_kLay(_G,G)  # next krim comparands
        if krim:
            if rng>1: G.kHH[-1] += [krim]  # kH = lays ( nodes
            else:     G.kHH = [[krim]]
            G_ += [G]
    Gd_ = copy(G_) # Gs with 1st layer kH, dLay, _kLay
    _G_ = G_; n=0  # n higher krims
    # convolution: kernel rim Def,Sum,Comp, in separate loops for bilateral G,_G assign:
    while True:
        G_ = []
        for G in _G_:  # += krim
            G.kHH[-1] += [[]]; G.visited__ += [[]]
        for G in _G_:
            #  append G.kHH[-1][-1]:
            for _G in G.kHH[-1][-2]:
                for link, rev in _G.rim_[-1]:
                    __G = link.nodet[0] if link.nodet[1] is G else link.nodet[1]
                    if __G in _G_:
                        if __G not in G.kHH[-1][-1] + [g for visited_ in G.visited__ for g in visited_]:
                            # bilateral add layer of unique mediated nodes
                            G.kHH[-1][-1] += [__G]; __G.kHH[-1][-1] += [G]
                            for g,_g in zip((G,__G),(__G,G)):
                                g.visited__[-1] += [_g]
                                if g not in G_:  # in G_ only if in visited__[-1]
                                    G_ += [g]
        for G in G_: G.visited__ += [[]]
        for G in G_: # sum kLay:
            for _G in G.kHH[-1][-1]:  # add last krim
                if _G in G.visited__[-1] or _G not in _G_:
                    continue  # Gs krim appended when _G was G
                G.visited__[-1] += [_G]; _G.visited__[-1] += [G]
                # sum alt G lower kLay:
                G.kLay = sum_kLay(G,_G); _G.kLay = sum_kLay(_G,G)
        for G in G_: G.visited__[-1] = []
        for G in G_:
            for _G in G.kHH[-1][0]:  # convo in direct kernel only
                if _G in G.visited__[-1] or _G not in G_: continue
                G.visited__[-1] += [_G]; _G.visited__[-1] += [G]
                # comp G kLay:
                dlay = comp_G(_G._kLay, G._kLay)
                if dlay.Et[0] > ave * dlay.Et[2] * (rng+n):  # layers add cost
                    dlay.node_ = G.kHH[-1][0]  # not sure we need it
                    _G.dLay.add_H(dlay); G.dLay.add_H(dlay)  # bilateral
        _G_ = G_; G_ = []
        for G in _G_:  # eval dLay
            G.visited__.pop()  # loop-specific layer
            if G.dLay.Et[0] > ave * G.dLay.Et[2] * (rng+n+1):
                G_ += [G]
        if G_:
            for G in G_: G._kLay = G.kLay  # comp in next krng
            _G_ = G_; n += 1
        else:
            for G in Gd_:
                if rng==1: G.extH.append_(CH().append_(G.dLay))
                else:      G.extH.H[-1].append_(G.dLay)
                delattr(G,'dLay'); delattr(G,'_kLay')
                if hasattr(G,"kLay)"): delattr(G,'kLay')
                G.visited__.pop()  # kH - specific layer
            break
    return Gd_, Et  # all Gs with dLay added in 1st krim

def sum_kLay(G, g):  # sum next-rng kLay from krim of current _kLays, init with G attrs

    KLay = (G.kLay if hasattr(G,"kLay")
                   else (G._kLay if hasattr(G,"_kLay")  # init conv kernels, also below:
                                 else (G.n,len(G.node_),G.S,G.A,deepcopy(G.latuple),CH().copy(G.mdLay),CH().copy(G.derH) if G.derH else None)))
    kLay = (G._kLay if hasattr(G,"_kLay")
                    else (g.n,len(g.node_),g.S,g.A,deepcopy(g.latuple),CH().copy(g.mdLay),CH().copy(g.derH) if g.derH else None))
                    # in init conv kernels
    N,L,S,A,Lat,MdLay,DerH = KLay
    n,l,s,a,lat,mdLay,derH = kLay
    return [
            N+n, L+l, S+s, [A[0]+a[0],A[1]+a[1]], # n,L,S,A
            add_lat(Lat,lat),                     # latuple
            MdLay.add_md_(mdLay),                 # mdLay
            DerH.add_H(derH) if derH else None ]  # derH

def rng_link_(_L_):  # comp CLs: der+'rng+ in root.link_ rim_t node rims: directional and node-mediated link tracing

    _mN_t_ = [[[L.nodet[0]],[L.nodet[1]]] for L in _L_]  # rim-mediating nodes
    rng = 1
    Et = [0,0,0,0]
    Link_, rL_ = [], []
    while True:
        mN_t_ = [[[],[]] for _ in _L_]  # for next loop
        for L, _mN_t, mN_t in zip(_L_, _mN_t_, mN_t_):
            for rev, _mN_, mN_ in zip((0,1), _mN_t, mN_t):
                # comp L, _Ls: nodet mN 1st rim, -> rng+ _Ls/ rng+ mm..Ns, flatten rim_s:
                rim_ = [[l for rim in n.rim_ for l in rim] if isinstance(n,CG) else n.rimt_[0][0] + n.rimt_[0][1] for n in _mN_]
                for rim in rim_:
                    for _L,_rev in rim:  # _L is reversed relative to its 2nd node
                        if _L is L or _L in L.visited_: continue
                        if not hasattr(_L,"rimt_"): add_der_attrs(link_=[_L])  # _L not in root.link_, same derivation
                        L.visited_ += [_L]; _L.visited_ += [L]
                        dy,dx = np.subtract(_L.yx, L.yx)
                        Link = CL(nodet=[_L,L], span=2, angle=[dy,dx], box=extend_box(_L.box, L.box))
                        # L.rim_t += new Link
                        if comp_N(Link, Et, rng, rev ^ _rev):  # negate ds if only one L is reversed
                            # L += rng+'mediating nodes, link orders: nodet < L < rimt_, mN.rim || L
                            Link_ += [Link]
                            mN_ += _L.nodet  # get _Ls in mN.rim
                            if _L not in _L_:
                                _L_ += [_L]; mN_t_ += [[[],[]]]  # not in root
                            elif _L not in rL_: rL_ += [_L]
                            if L not in rL_:    rL_ += [L]
                            mN_t_[_L_.index(_L)][1 - rev] += L.nodet
                            for node in (L, _L):
                                if len(node.extH.H) < rng:
                                    node.extH.append_(Link.derH)
                                else: node.extH.H[-1].add_H(Link.derH)
        L_, mN_t_ = [],[]
        for L, mN_t in zip(_L_, mN_t_):
            if any(mN_t):
                L_ += [L]; _mN_t_ += [mN_t]
        if L_:
            _L_ = L_; rng += 1
        else:
            break
        # Lt_ = [(L, mN_t) for L, mN_t in zip(L_, mN_t_) if any(mN_t)]; if Lt_: L_,_mN_t_ = map(list, zip(*Lt_))  # map list to convert tuple from zip(*)

    return list(set(rL_)), Link_, Et, rng


def comp_N(Link, iEt, rng, rev=None):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = rev is not None  # compared links have binary relative direction?
    _N, N = Link.nodet; rn = _N.n/N.n
    # no comp extH, it's current ders
    if fd:  # CLs
        DLay = _N.derH.comp_H(N.derH, rn, fagg=1)  # new link derH = local dH
        _A,A = _N.angle, N.angle if rev else [-d for d in N.angle] # reverse if left link
    else:   # CGs
        DLay = comp_G([_N.n,len(_N.node_),_N.S,_N.A,_N.latuple,_N.mdLay,_N.derH],
                      [N.n, len(N.node_), N.S, N.A, N.latuple, N.mdLay, N.derH])
        for H in DLay.H: H.node_ += [_N, N]
        DLay.root = Link
        _A,A = _N.A,N.A
    DLay.node_ = [_N,N]
    Link.mdext = comp_ext(2,2, _N.S,N.S/rn, _A,A)
    '''
    if fd: Link.derH.append_(DLay)
    else:  Link.derH.append(DLay, flat=1)  # merge
    '''
    Link.derH.append_(DLay, flat=1-fd)  # same as above
    iEt[:] = np.add(iEt,DLay.Et)  # init eval rng+ and form_graph_t by total m|d?
    for i in 0,1:
        Val, Rdn = DLay.Et[i::2]
        if Val > G_aves[i] * Rdn * (rng+1): Link.ft[i] = 1  # fork inclusion tuple
        _N.Et[i] += Val; N.Et[i] += Val  # not selective
        _N.Et[2+i] += Rdn; N.Et[2+i] += Rdn  # per fork link in both Gs
        # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dH.Et[i::2])]
    if any(Link.ft):
        Link.n = min(_N.n,N.n)  # comp shared layers
        Link.yx = np.add(_N.yx, N.yx) / 2
        Link.S += (_N.S + N.S) / 2
        for rev, node in zip((0,1),(_N,N)):  # ?reversed Link direction
            if fd:
                if len(node.rimt_) == rng:
                    node.rimt_[-1][1-rev] += [[Link,rev]]  # add in last rng layer, opposite to _N,N dir
                else: node.rimt_ += [[[[Link,rev]],[]]] if dir else [[[],[[Link,rev]]]]  # add rng layer
            else:
                if len(node.extH.H) == rng:
                    node.rim_[-1] += [[Link, rev]]  # accum last rng layer
                else: node.rim_ += [[[Link, rev]]]  # init rng layer

        return True

def comp_G(_pars, pars):  # compare kLays or partial graphs in merging

    _n,_L,_S,_A,_latuple,_mdLay,_derH = _pars
    n, L,S,A, latuple,mdLay,derH = pars; rn = _n/n

    mdext = comp_ext(_L,L,_S,S/rn,_A,A)
    if mdLay:  # += CG nodes
        mdLay = _mdLay.comp_md_(mdLay, rn, fagg=1)
        mdlat = comp_latuple(_latuple, latuple, rn, fagg=1)
        n = mdlat.n + mdLay.n + mdext.n
        md_t = [mdlat, mdLay, mdext]
        Et = np.array(mdlat.Et) + mdLay.Et + mdext.Et
        Rt = np.array(mdlat.Rt) + mdLay.Rt + mdext.Rt
    else:  # += CL nodes
        n = mdext.n; md_t = [mdext]; Et = mdext.Et; Rt = mdext.Rt
    # single-layer H:
    derH = CH( H=[CH(n=n,md_t=md_t,Et=Et,Rt=Rt)], n=n, md_t=[CH().copy(md_) for md_ in md_t], Et=copy(Et), Rt=copy(Rt))
    if _derH and derH:
        dderH = _derH.comp_H(derH, rn, fagg=1)  # new link derH = local dH
        derH.append_(dderH, flat=1)

    return derH

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


def form_graph_t(root, N_, L_, Et, rng):  # segment N_ to Nm_, Nd_

    node_t = []
    for fd in 0,1:
        # N_ is Link_ if fd else G_
        if Et[fd] > ave * Et[2+fd]:
            if isinstance(N_[0],CG):  # link.root is empty?
                 for G in N_: G.root = []  # new root will be intermediate graph
            # G: Ls/D if fd: directional, else Ns/M: symmetric, sum in node:
            graph_ = segment_N_(root, L_ if fd else N_, fd, rng)
            for graph in graph_:
                Q = graph.node_  # xcomp -> max_dist * rng+1  (always graph.node_ now? Since when fd == 1, nodes will be links
                if len(Q) > ave_L and graph.derH.Et[fd] > G_aves[fd] * graph.derH.Et[fd+2]:
                    if fd: add_der_attrs(Q)
                    agg_recursion(graph, Q, fL=isinstance(Q[0],CL), rng=rng)  # fd rng+
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]
    for fd, graph_ in enumerate(node_t):  # combined-fork fb
        for graph in graph_:
            if graph.derH:  # single G's graph doesn't have any derH.H to feedback
                root.fback_t[fd] += [graph.derH] if fd else [graph.derH.H[-1]]  # der+ forms new links, rng+ adds new layer
            # sub+-> sub root-> init root
    if any(root.fback_t): feedback(root)

    return node_t

def add_der_attrs(link_):
    for link in link_:
        link.extH = CH()  # for der+
        link.root = None  # dgraphs containing link
        link.rimt_ = [[[],[]]]  # dirt per rng layer
        link.visited_ = []
        link.med = 1  # comp med rng, replaces len rim_
        link.Et = [0,0,0,0]
        link.aRad = 0

def segment_N_(root, iN_, fd, rng):  # form proto sub-graphs in iN_: Link_ if fd else G_
    '''
    Cluster by weight of shared links, initially single linkage.
    Link clustering (der+) is secondary and node-mediated, contained in a graph formed by these links.
    That makes link clustering a divisive phase of agglomerative node clustering (agg+), constraining search space.
    '''
    N_ = []
    max_ = []
    for N in iN_:   # init Gt per N:
        # mediator: iN if fd else iL
        med_ = N.nodet if fd else [Lt[0] for Lt in (N.rim_[-1] if isinstance(N,CG) else N.rimt_[-1][0] + N.rimt_[-1][1])]
        if fd:  # get links of nodet
            eN_ = [Lt[0] for _N in med_ for Lt in (_N.rim_[-1] if isinstance(_N,CG) else _N.rimt_[-1][0]+_N.rimt_[-1][1])]
        else:   # get nodes of rim links
            eN_ = [_N for _L in med_ for _N in _L.nodet]
        Nrim = [e for e in eN_ if e is not N and e in iN_]  # Ns connected by Gt-external links, not in node_ yet
        Gt = [[N], [], copy(med_), Nrim, [0,0,0,0]]  # [node_, link_, Lrim, Nrim, Et]
        N.root = Gt; N_ += [Gt]
        emax_ = []  # if exemplar G|Ls:
        for eN in Nrim:
            eEt,Et = (eN.derH.Et, N.derH.Et) if fd else (eN.extH.Et, N.extH.Et)
            if eEt[fd] >= Et[fd] or eN in max_: emax_ += [eN]  # _N if _N == N
        if not emax_: max_ += [N]  # no higher-val neighbors
        # add rrim mediation: V * k * max_rng?
    max_ = [N.root for N in max_]
    for Gt in N_: Gt[3] = [_N.root for _N in Gt[3]]  # replace ext Ns with their Gts
    for Gt in max_ if max_ else N_:
        node_, link_, Lrim, Nrim, Et = Gt
        for _Gt, _L in zip(Nrim, Lrim):  # merge connected _Gts if >ave shared external links (Lrim)
            if _Gt not in N_:
                continue  # was merged
            oL_ = set(Lrim).intersection(set(_Gt[2])).union([_L])  # shared external links + potential _L, or oL_ = [Lr[0] for Lr in _Gt[2] if Lr in Lrim]
            if isinstance(_L, CL):  # link is CL
                oV = sum([L.derH.Et[fd] - ave * L.derH.Et[2+fd] for L in oL_])
            else:  # link is CG
                oV = sum([L.extH.Et[fd] - ave * L.extH.Et[2+fd] for L in oL_])
            if oV > ave:
                link_ += [_L]
                merge(Gt,_Gt)
                N_.remove(_Gt)

    return [sum2graph(root, Gt, fd, rng) for Gt in N_]

def merge(Gt, gt):

    N_,L_, Lrim, Nrim, Et = Gt
    n_,l_, lrim, nrim, et = gt
    N_ += n_
    L_ += l_  # internal, no overlap
    Lrim[:] = list(set(Lrim + lrim))  # exclude shared external links, direction doesn't matter?
    Nrim[:] += [root for root in nrim if root not in Nrim]  # Nrim contains root of rim now
    Et[:] = np.add(Et,et)

def sum2graph(root, grapht, fd, rng):  # sum node and link params into graph, aggH in agg+ or player in sub+

    N_, L_, _,_, Et = grapht  # [node_, link_, Lrim, Nrim_t, Et]
    graph = CG(fd=fd, node_=N_, link_=L_, rng=rng, Et=Et)
    graph.root = root
    yx = [0,0]
    for N in N_:
        graph.area += N.area
        graph.box = extend_box(graph.box, N.box)
        if isinstance(N,CG):
            graph.mdLay.add_md_(N.mdLay)
            add_lat(graph.latuple, N.latuple)
        graph.n += N.n  # non-derH accumulation?
        if N.derH: graph.derH.add_H(N.derH)
        N.root = graph
        yx = np.add(yx, N.yx)
    L = len(N_)
    yx = np.divide(yx,L); graph.yx = yx
    # ave distance from graph center to node centers:
    graph.aRad = sum([np.hypot(*np.subtract(yx,N.yx)) for N in N_]) / L
    extH = CH(node_ = copy(N_))  # should be a different list reference here
    
    if L_:
        if isinstance(L_[0], CL):  # CL
            for link in L_:  # unique current-layer links
                if isinstance(link, CL):
                    graph.S += link.span
                    graph.A = np.add(graph.A,link.angle)  # np.add(graph.A, [-link.angle[0],-link.angle[1]] if rev else link.angle)
                    extH.add_H(link.derH) if extH else extH.append_(link.derH, flat=1)
        else:  # CG
            link_ = list(set([Lt[0] for G in L_ for rim in G.rim_ for Lt in rim]))  # remove overlapping links
            for link in link_: extH.add_H(link.derH) if extH else extH.append_(link.derH, flat=1)
            graph.S += sum([G.S for G in L_])
            for G in L_: graph.A = np.add(graph.A,G.A)
         
    if extH: graph.derH.append_(extH, flat=0)
    if fd:
        # assign alt graphs from d graph, after both linked m and d graphs are formed
        for node in graph.node_:  # we assign root in node now, and node can be CG or CL
            mgraph = node.root
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]
    return graph

def feedback(root):  # called from form_graph_, always sub+, append new der layers to root

    mDerLay = CH()  # added per rng+
    while root.fback_t[0]:
        mDerLay.add_H(root.fback_t[0].pop())
    dDerH = CH()  # from higher-order links
    while root.fback_t[1]:
        dDerH.add_H(root.fback_t[1].pop())
    DderH = mDerLay.append_(dDerH, flat=1)
    m,d, mr,dr = DderH.Et
    if m+d > sum(G_aves) * (mr+dr):
        root.derH.append_(DderH, flat=0)  # new derLay maybe nested