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
Clustering criterion is also M|D, summed across >ave vars if selective comp (<ave vars are not compared and don't add costs).
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

    def __init__(l, nodet=None,derH=None, S=0, A=None, box=None, md_t=None, H_=None):
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
        l.root_ = None
        # rimt_, elay if der+
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
                            PP.elay = CH()  # init empty per agg+
                    if len(pruned_node_) > 10:  # discontinuous PP rng+ cross-comp, cluster -> G_t:
                        agg_recursion(edge, iaggH=[[pruned_node_,[],edge.Et]])
                        # updates edge node_,link_

def agg_recursion(root, iaggH):  # top lay rng++-> two cluster,agg++ forks per rng layer:

    iN_,iL_,iEt = iaggH[0]  # top input aggLay, same syntax for all lays
    # rng++ cross-comp:
    rngH, L_,Et = rng_node_(root, iN_) if isinstance(iN_[0],CG) else rng_link_(root, iN_)
    # agg++ nesting in L_ is lower than in N_?
    root.link_ = L_
    for N in iN_:
        for fd in 0,1:  # in rng++ elay: fd derR -> valR:
            for i, lay in enumerate(sorted(N.elay.H, key=lambda lay: lay.Et[fd], reverse=True)):
                di = lay.i - i  # lay.i: index in H
                lay.Et[2+fd] += di  # derR-valR
                if not i:  # max value lay
                    N.node_ = lay.node_; N.derH.it[fd] = lay.i  # assigns exemplar lay index per fd
    hEt = [0,0,0,0]
    for rng, rLay in enumerate(rngH, start=1):  # replace rngLay' N_,L_ with aggHs (rng starts with 1 instead of 0)
        nG_,lG_,rL_,rEt = rLay  # rL_ probably is not needed
        for fd, G_ in zip((0,1),(nG_,lG_)):
            AEt = rEt  # val agg++/_G_ (rename for clarity? Else i don't see why we need reassign a new var with same reference)
            aggH = [[G_,[],rEt]]
            while len(G_) > ave_L and AEt[fd] > G_aves[fd] * AEt[2+fd] * rng:
                G_H = agg_recursion(root,aggH)  # replaces G_ with G_H, empty L_: always lags behind N_?
                if G_H:
                    nH, lH, aEt = G_H[0]  # aggLay added to rngLay
                    if aEt[0] > G_aves[0] * aEt[2]:
                        AEt = np.add(AEt,aEt)
                        aggH[:] = G_H  # replaces nG_|lG_ in rLay,rngH
                else: break
            rEt[:] = AEt  # rng++ + agg++ vals
            hEt = np.add(hEt,AEt)  # from both forks
    Et = np.add(Et,hEt)
    # eval rngH, nested by agg++:
    if sum(Et[:1]) > sum(G_aves) * sum(Et[2:]):
        iEt[:] = np.add(iEt,Et)
        # return appendleft aggH:
        # this rngH is layered, so return only top layer N_?
        return [[rngH[0][0],L_,Et]] + iaggH
'''
    feed back new layers
    for fd, graph_ in enumerate(node_t):  # combined-fork fb
        for graph in graph_:
            if graph.derH:  # single G's graph doesn't have any derH.H to feedback
                root.fback_t[fd] += [graph.derH] if fd else [graph.derH.H[-1]]  # der+ forms new links, rng+ adds new layer
            # sub+-> sub root-> init root
    if any(root.fback_t): feedback(root)
'''

def rng_node_(root,_N_):  # each rng+ forms rim_ layer per N, then nG_,L_,dG_,Et:

    rngH, L__ = [],[]; HEt = [0,0,0,0]
    rng = 1
    while True:
        N_,Et = rng_kern_(_N_,rng)  # adds a layer of links to _Ns with pars summed in G.kHH[-1]
        mG_ = segment_N_(root, N_,0,rng)  # cluster N_ by link M
        L_  = [link for G in mG_ for link in G.link_]
        dG_ = segment_N_(root, L_,1,rng)  # cluster L__ by D (no sum/N: direction is lost and D is redundant to M)
        set_attrs(L_)  # reset L_ after segment? Else their Et will be reset to zero
        rngH += [[mG_,dG_,L_,Et]]
        L__+= L_
        HEt = np.add(HEt, Et)
        if Et[0] > ave * Et[2] * rng:
            _N_ = N_; rng += 1
        else:
            break
    return rngH, L_, HEt

def rng_kern_(N_, rng):  # comp Gs summed in kernels, ~ graph CNN without backprop, not for CLs

    _G_ = []; Et = [0,0,0,0]
    # comp_N:
    for _G,G in combinations(N_,r=2):
        if _G in [g for visited_ in G.visited__ for g in visited_]:  # compared in any rng++
            continue
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        aRad = (G.aRad+_G.aRad) / 2  # ave G radius
        # eval relative distance between G centers:
        if dist / max(aRad,1) <= (max_dist * rng):
            for _g,g in (_G,G),(G,_G):
                if len(g.visited__) == rng:
                    g.visited__[-1] += [_g]
                else: g.visited__ += [[_g]]  # init layer
            Link = CL(nodet=[_G,G], S=2,A=[dy,dx], box=extend_box(G.box,_G.box))
            if comp_N(Link, Et, rng):
                for g in _G,G:
                    if g not in _G_: _G_ += [g]
    G_ = []  # init conv kernels:
    for G in (_G_):
        krim = []
        for link,rev in G.rim_[-1]:  # form new krim from current-rng links
            if link.ft[0]:  # must be mlink
                _G = link.nodet[0] if link.nodet[1] is G else link.nodet[1]
                krim += [_G]
                G.elay.add_H(link.derH)
                G._kLay = sum_kLay(G,_G); _G._kLay = sum_kLay(_G,G)  # next krim comparands
        if krim:
            if rng>1: G.kHH[-1] += [krim]  # kH = lays(nodes
            else:     G.kHH += [[krim]]
            G_ += [G]
    Gd_ = copy(G_)  # Gs with 1st layer kH, dLay, _kLay
    _G_ = G_; n=0  # n higher krims
    # convolution: Def,Sum,Comp kernel rim, in separate loops for bilateral G,_G assign:
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
                # comp G kLay -> rng derLay:
                rlay = comp_pars(_G._kLay, G._kLay, _G.n/G.n)  # or/and _G.kHH[-1][-1] + G.kHH[-1][-1]?
                if rlay.Et[0] > ave * rlay.Et[2] * (rng+n):  # layers add cost
                    _G.elay.add_H(rlay); G.elay.add_H(rlay)  # bilateral
        _G_ = G_; G_ = []
        for G in _G_:  # eval dLay
            G.visited__.pop()  # loop-specific layer
            if G.elay.Et[0] > ave * G.elay.Et[2] * (rng+n+1):
                G_ += [G]
        if G_:
            for G in G_: G._kLay = G.kLay  # comp in next krng
            _G_ = G_; n += 1
        else:
            for G in Gd_:
                G.visited__.pop()  # kH - specific layer
                delattr(G,'_kLay')
                if hasattr(G,'kLay'): delattr(G,'kLay')
            break

    return Gd_, Et  # all Gs with dLay added in 1st krim

def sum_kLay(G, g):  # sum next-rng kLay from krim of current _kLays, init with G attrs

    KLay = (G.kLay if hasattr(G,"kLay")
                   else (G._kLay if hasattr(G,"_kLay")  # init conv kernels, also below:
                              else (len(G.node_),G.S,G.A,deepcopy(G.latuple),CH().copy(G.mdLay),CH().copy(G.derH) if G.derH else CH())))  # init DerH if empty
    kLay = (G._kLay if hasattr(G,"_kLay")
                    else (len(g.node_),g.S,g.A,deepcopy(g.latuple),CH().copy(g.mdLay),CH().copy(g.derH) if g.derH else None))  # in init conv kernels
    L,S,A,Lat,MdLay,DerH = KLay
    l,s,a,lat,mdLay,derH = kLay
    return [
            L+l, S+s, [A[0]+a[0],A[1]+a[1]], # L,S,A
            add_lat(Lat,lat),                # latuple
            MdLay.add_md_(mdLay),            # mdLay
            DerH.add_H(derH) if derH else DerH
    ]

def rng_link_(root, _L_):  # comp CLs: der+'rng+ in root.link_ rim_t node rims: directional and node-mediated link tracing

    _mN_t_ = [[[L.nodet[0]],[L.nodet[1]]] for L in _L_]  # rim-mediating nodes in both directions
    rH = []; HEt = [0,0,0,0]
    rng = 1
    while True:
        Et = [0,0,0,0]
        mN_t_ = [[[],[]] for _ in _L_]  # new rng lay of mediating nodes, traced from all prior layers?
        for L, _mN_t, mN_t in zip(_L_, _mN_t_, mN_t_):
            for rev, _mN_, mN_ in zip((0,1), _mN_t, mN_t):
                # comp L, _Ls: nodet mN 1st rim, -> rng+ _Ls/ rng+ mm..Ns, flatten rim_s:
                rim_ = [rim for n in _mN_ for rim in (n.rim_ if isinstance(n, CG) else [n.rimt_[0][0] + n.rimt_[0][1]])]
                for rim in rim_:
                    for _L,_rev in rim:  # _L is reversed relative to its 2nd node
                        if _L is L or _L in L.visited_: continue
                        if not hasattr(_L,"rimt_"): set_attrs([_L])  # _L not in root.link_, same derivation
                        L.visited_ += [_L]; _L.visited_ += [L]
                        dy,dx = np.subtract(_L.yx, L.yx)
                        Link = CL(nodet=[_L,L], S=2, A=[dy,dx], box=extend_box(_L.box, L.box))
                        # L.rim_t += new Link
                        if comp_N(Link, Et, rng, rev ^ _rev):  # negate ds if only one L is reversed
                            # L += rng+'mediating nodes, link orders: nodet < L < rimt_, mN.rim || L
                            mN_ += _L.nodet  # get _Ls in mN.rim
                            if _L not in _L_:
                                _L_ += [_L]; mN_t_ += [[[],[]]]  # not in root
                            mN_t_[_L_.index(_L)][1 - rev] += L.nodet
                            for node in (L, _L):
                                node.elay.add_H(Link.derH)
        # cluster by rim M:
        graph_ = segment_N_(root, _L_, 0, rng)
        if graph_:
            rH += [[graph_,Et]]
            HEt = np.add(HEt,Et)
        V = 0; L_,_mN_t_ = [],[]
        for L, mN_t in zip(_L_,mN_t_):
            if any(mN_t):
                L_ += [L]; _mN_t_ += [mN_t]
                V += L.derH.Et[0] - ave * L.derH.Et[2] * rng
        if V > 0:  # rng+ if vM of Ls with extended mN_t_
            _L_ = L_; rng += 1
        else:
            break
    return rH, HEt


def comp_N(Link, iEt, rng, rev=None):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = rev is not None  # compared links have binary relative direction?
    _N, N = Link.nodet; _S,S = _N.S,N.S; _A,A = _N.A,N.A
    if fd:  # CL
        if rev: A = [-d for d in A]  # reverse angle direction if N is left link?
        _L=2; L=2; _lat,lat,_lay,lay = None,None,None,None
    else:   # CG
        _L,L,_lat,lat,_lay,lay = len(_N.node_),len(_N.node_),_N.latuple,N.latuple,_N.mdLay,N.mdLay
    # dlay:
    derH = comp_pars([_L,_S,_A,_lat,_lay,_N.derH], [L,S,A,lat,lay,N.derH], rn=_N.n/N.n)
    derH.root = Link; Link.Et = derH.Et[:]  # looks like this is missed out
    Et = derH.Et
    iEt[:] = np.add(iEt,Et)  # init eval rng+ and form_graph_t by total m|d?
    for i in 0,1:
        Val, Rdn = Et[i::2]
        if Val > G_aves[i] * Rdn * (rng+1): Link.ft[i] = 1  # fork inclusion tuple
        _N.Et[i] += Val; N.Et[i] += Val  # not selective
        _N.Et[2+i] += Rdn; N.Et[2+i] += Rdn  # per fork link in both Gs
        # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dH.Et[i::2])]
    if any(Link.ft):
        Link.derH = derH; derH.root = Link; Link.Et = Et; Link.n = min(_N.n,N.n)  # comp shared layers
        Link.nodet = [_N,N]; Link.yx = np.add(_N.yx,N.yx) /2
        # S,A set before comp_N
        for rev, node in zip((0,1),(_N,N)):  # ?reversed Link direction
            if fd:
                if len(node.rimt_)==rng: node.rimt_[-1][1-rev] += [[Link,rev]]  # add in last rng layer, opposite to _N,N dir
                else:                    node.rimt_ = [[[[Link,rev]],[]]] if dir else [[[],[[Link,rev]]]]  # add rng layer
            else:
                if len(node.rim_)==rng: node.rim_[-1] += [[Link, rev]]
                else:                   node.rim_ += [[[Link, rev]]]
            # elay += derH in rng_kern_
        return True

def comp_pars(_pars, pars, rn):  # compare Ns, kLays or partial graphs in merging

    _L,_S,_A,_latuple,_mdLay,_derH = _pars
    L, S, A, latuple, mdLay, derH = pars

    mdext = comp_ext(_L,L,_S,S/rn,_A,A)
    if mdLay:  # CG
        mdLay = _mdLay.comp_md_(mdLay, rn, fagg=1)
        mdlat = comp_latuple(_latuple, latuple, rn, fagg=1)
        n = mdlat.n + mdLay.n + mdext.n
        md_t = [mdlat, mdLay, mdext]
        Et = np.array(mdlat.Et) + mdLay.Et + mdext.Et
        Rt = np.array(mdlat.Rt) + mdLay.Rt + mdext.Rt
    else:  # += CL nodes
        n = mdext.n; md_t = [mdext]; Et = mdext.Et; Rt = mdext.Rt
    # init H = [lay0]:
    dlay = CH( H=[CH(n=n, md_t=md_t, Et=Et, Rt=Rt)],  # or n = (_n+n) / 2?
               n=n, md_t=[CH().copy(md_) for md_ in md_t], Et=copy(Et),Rt=copy(Rt))
    if _derH and derH:
        dderH = _derH.comp_H(derH, rn, fagg=1)  # new link derH = local dH
        dlay.append_(dderH, flat=1)

    return dlay

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

def set_attrs(Q):

    for e in Q:
        e.visited_ = []
        if isinstance(e, CL):
            e.rimt_ = [[[],[]]]  # nodet-mediated links, same der order as e
            e.root_ = []  # or root_ should be default param in CL too?
        if hasattr(e,'elay'): e.derH.append_(e.elay)  # no default CL.elay
        e.elay = CH()  # set in sum2graph
        e.Et = [0,0,0,0]
        e.aRad = 0
    return Q

def segment_N_(root, iN_, fd, rng):  # cluster iN_(G_|L_) by weight of shared links, initially single linkage

    max_, N_ = [],[]
    for N in iN_:  # init Gt per G|L node:
        if fd:
            if isinstance(N.nodet[0],CG):
                  Nrim = [Lt[0] for n in N.nodet for Lt in n.rim_[-1] if (Lt[0] is not N and Lt[0] in iN_)]  # external nodes
            else: Nrim = [Lt[0] for Lt in N.rimt_[-1][0]+ N.rimt_[-1][1] if (Lt[0] is not N and Lt[0] in iN_)]  # nodet-mediated links, same der order as N
            Lrim = Nrim  # mediated CL connection, no links (maybe simplified)
            N.root_ = []  # init
        else:
            Lrim = [Lt[0] for Lt in N.rim_[-1]] if isinstance(N,CG) else [Lt[0] for Lt in N.rimt_[-1][0] + N.rimt_[-1][1]]  # external links
            Nrim = [_N for L in Lrim for _N in L.nodet if (_N is not N and _N in iN_)]  # external nodes
        Gt = [[N],[], Lrim, Nrim, [0,0,0,0]]  # node_,link_,Lrim,Nrim, Et
        N.root_+= [Gt]
        N_ += [Gt]  # select exemplar maxes to segment clustering:
        emax_ = [eN for eN in Nrim if eN.Et[fd] >= N.Et[fd] or eN in max_]  # _N if _N == N
        if not emax_: max_ += [Gt]  # N.root, if no higher-val neighbors
        # extended rrim max: V * k * max_rng?
    for Gt in N_: Gt[3] = [_N.root_[-1] for _N in Gt[3]]  # replace eNs with Gts
    # merge Gts with shared +ve links:
    for Gt in max_ if max_ else N_:
        node_,link_, Lrim, Nrim, Et = Gt
        for _Gt, _L in zip(Nrim,Lrim):  # merge connected _Gts if >ave shared external links (Lrim)
            if _Gt not in N_: continue  # was merged
            # for fd == 1, looks like we need to get root[0][0] (first link) since Nrim is replace by root
            sL_ = set([root[0][0] for root in Nrim] if fd else Lrim).intersection(set([root[0][0] for root in _Gt[3]] if fd else _Gt[2] )).union([_L])  # or oL_ = [Lr[0] for Lr in _Gt[2] if Lr in Lrim]
            # shared external links if fd else nodes, + potential _L|N_
            Et = np.sum([L.Et for L in sL_],axis=0)
            if Et[fd] > ave * Et[2+fd] * rng > 0:  # value of shared links or nodes
                if not fd: link_ += [_L]
                merge(Gt,_Gt)
                N_.remove(_Gt)
    return [sum2graph(root, Gt, fd, rng) for Gt in N_]

def merge(Gt, gt):

    N_,L_, Lrim, Nrim, Et = Gt
    n_,l_, lrim, nrim, et = gt
    N_ += n_
    for N in N_: N.root_[-1] = Gt
    L_ += l_  # internal, no overlap
    Lrim[:] = list(set(Lrim + lrim))  # exclude shared external links, direction doesn't matter?
    Nrim[:] += [root for root in nrim if root not in Nrim and root is not Gt]  # Nrim is roots of ext Ns, may also be in other Gts Nrim
    Et[:] = np.add(Et,et)

def sum2graph(root, grapht, fd, rng):  # sum node and link params into graph, aggH in agg+ or player in sub+

    N_, L_, _,_, Et = grapht  # [node_, link_, Lrim, Nrim_t, Et]
    graph = CG(fd=fd, root_ = [root], node_=N_, link_=L_, rng=rng, Et=Et)
    yx = [0,0]
    lay0 = CH(node_= N_)  # comparands, vs. L_: summands?
    for link in L_:  # unique current-layer mediators: Ns if fd else Ls
        graph.S += link.S
        graph.A = np.add(graph.A,link.A)  # np.add(graph.A, [-link.angle[0],-link.angle[1]] if rev else link.angle)
        lay0.add_H(link.derH) if lay0 else lay0.append_(link.derH)
    graph.derH.append_(lay0)  # empty for single-node graph
    derH = CH()
    for N in N_:
        graph.area += N.area
        graph.box = extend_box(graph.box, N.box)
        if isinstance(N,CG):
            graph.mdLay.add_md_(N.mdLay)
            add_lat(graph.latuple, N.latuple)
        graph.n += N.n  # +derH.n
        if N.derH: derH.add_H(N.derH)
        N.root_ += [graph]
        yx = np.add(yx, N.yx)
    graph.derH.append_(derH, flat=1)  # comp(derH) forms new layer, higher layers are added by feedback
    L = len(N_)
    yx = np.divide(yx,L); graph.yx = yx
    # ave distance from graph center to node centers:
    graph.aRad = sum([np.hypot(*np.subtract(yx,N.yx)) for N in N_]) / L
    if fd:
        # assign alt graphs from d graph, after both linked m and d graphs are formed
        for node in graph.node_:  # CG or CL
            mgraph = node.root_[-1]
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
        root.derH.append_(DderH, flat=1)  # multiple lays appended upward