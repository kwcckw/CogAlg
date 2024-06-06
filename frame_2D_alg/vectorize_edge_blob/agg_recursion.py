import numpy as np
from copy import deepcopy, copy
from itertools import combinations, product, zip_longest
from .slice_edge import comp_angle, CsliceEdge
from .comp_slice import ider_recursion, comp_latuple, get_match, CH, CG, CdP
from utils import box2center, extend_box
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

class Clink(CBase):  # product of comparison between two nodes or links
    name = "link"

    def __init__(l, nodet=None,rim=None, derH=None, extH=None, root=None, distance=0, angle=None, box=None ):
        super().__init__()
        # Clink = Ltree: binary tree of Gs,
        # Ltree depth+ / der+: Clink is 2 connected Gs, Clink with Clink nodet is 4 connected Gs, etc, unpack sequentially.
        l.nodet = [] if nodet is None else nodet  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.n = 1  # min(node_.n)
        l.S = 0  # sum nodet
        l.angle = [0,0] if angle is None else angle  # dy,dx between node centers
        l.distance = distance  # distance between node centers
        l.box = [] if box is None else box  # sum nodet
        l.area = 0  # sum nodet
        l.latuple = []  # sum nodet
        l.Vt = [0,0]  # for rim-overlap modulated segmentation, init derH.Et[:2]
        l.derH = CH() if derH is None else derH
        l.DerH = CH()  # ders from kernels: G.DerH

    def __bool__(l): return bool(l.derH.H)


def vectorize_root(image):  # vectorization in 3 composition levels of xcomp, cluster:

    frame = CsliceEdge(image).segment()
    for edge in frame.blob_:
        if hasattr(edge, 'P_') and edge.latuple[-1] * (len(edge.P_)-1) > G_aves[0]:  # eval G, rdn=1
            edge.derH = CH(); edge.iderH = CH(); edge.fback_=[]; edge.Et=[0,0,0,0]; edge.link_=[]
            for P, _ in edge.P_:
                P.derH = CH()
            ider_recursion(None, edge)  # vertical, lateral-overlap P cross-comp -> PP clustering
            node_t, link_t = [[],[]], [[],[]]
            for fd, node_ in enumerate(copy(edge.node_)):  # always node_t
                if edge.iderH and any(edge.iderH.Et):  # any for np array
                    if edge.iderH.Et[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.iderH.Et[2+fd]:
                        pruned_node_ = []
                        for PP in node_:  # PP -> G
                            if PP.iderH and PP.iderH.Et[fd] > G_aves[fd] * PP.iderH.Et[2+fd]:  # v> ave*r
                                PP.root_ = []  # no feedback to edge?
                                PP.node_ = PP.P_  # revert base node_
                                PP.Et = [0,0,0,0]  # [] in comp_slice
                                pruned_node_ += [PP]
                        if len(pruned_node_) > 10:  # discontinuous PP rng+ cross-comp, cluster -> G_t:
                            node_t[fd] = pruned_node_
                            agg_recursion(None, edge, N_=pruned_node_, fagg=1)
                            link_t[fd] = edge.link_
            if any(node_t):
                edge.node_ = node_t; edge.link_ = link_t

def agg_recursion(rroot, root, N_, rng=1, fagg=0):  # rng for sub+'rng+ only

    Et = [0,0,0,0]
    # rng+: comp nodes|links mediated by previously compared N|Ls -> N_|L_:
    N_,rng, Et = rng_convolve(N_,Et,rng) if fagg else rng_trace_rim(N_,Et)

    node_t = form_graph_t(root, N_, Et, rng)  # sub+ and feedback
    if node_t:
        for fd, node_ in enumerate(node_t):
            N_ = [n for n in node_ if n.derH.Et[0] > G_aves[fd] * n.derH.Et[2]]  # pruned node_
            # comp val is proportional to n comparands:
            if root.derH.Et[0] * ((len(N_)-1)*root.rng) > G_aves[1]*root.derH.Et[2]:
                # agg+ / node_t, vs. sub+ / node_, always rng+:
                agg_recursion(rroot, root, N_, fagg=1)
                if rroot and root.derH:
                    rroot.fback_ += [root.derH]
                    feedback(rroot, fd)  # update root.root..
        root.node_[:] = node_t  # else keep root.node_


def rng_convolve(N_, Et, rng=1):  # comp Gs|kernels in agg+, links | link rim_t node rims in sub+
                                  # similar to graph convolutional network but without backprop
    _G_ = N_
    G_ = []  # eval comp_N -> G_:
    for link in list(combinations(_G_,r=2)):
        _G, G = link
        if _G in G.compared_: continue
        cy, cx = box2center(G.box); _cy, _cx = box2center(_G.box)
        dy = cy-_cy; dx = cx-_cx;  dist = np.hypot(dy,dx)
        if dist <= max_dist *rng:  # eval distance between node centers
            G.compared_ += [_G]; _G.compared_ += [G]
            Link = Clink(nodet=[_G,G], distance=dist, angle=[dy,dx], box=extend_box(G.box,_G.box))
            comp_N(Link, G_, Et)
    _G_ = list(set(G_))
    for G in _G_:  # init kernel as [krim]
        krim = []
        for link in G.rim:
            if G.derH: G.derH.add_(link.derH)
            else: G.derH = deepcopy(link.derH)
            krim += [link.nodet[0] if link.nodet[1] is G else link.nodet[1]]
        G.kH = [krim]
    rng = 1  # aggregate rng+: recursive center node DerH += linked node derHs for next-loop cross-comp
    while len(_G_) > 2:
        G_ = []
        for G in _G_:
            if len(G.rim) < 2: continue  # one link is always overlapped
            for link in G.rim:
                if link.derH.Et[0] > ave:  # link.Et+ per rng
                    comp_krim(link, G_, rng)  # + kernel rim / loop, sum in G.extH, derivatives in link.extH?
        _G_ = list(set(G_))
        rng += 1
    for G in _G_:
        for i, link in enumerate(G.rim):
            G.extH.add_(link.DerH) if i else G.extH.append_(link.DerH, flat=1)  # for segmentation

    return N_, rng, Et

def rng_trace_rim(N_, Et):  # comp Clinks: der+'rng+ in root.link_ rim_t node rims: directional and node-mediated link tracing

    _L_ = N_
    rng = 1
    while _L_:
        L_ = []
        for L in _L_:
            for dir, prerim in zip((0,1), L.prerimt):  # Link direction
                for _L in prerim:
                    if _L is L or _L in L.compared_: continue
                    if not hasattr(_L,"rimt_"): add_der_attrs( link_=[_L])  # _L is outside root.link_, same derivation
                    L.compared_ += [_L]
                    _L.compared_ += [L]
                    Link = Clink(nodet =[_L,L], box=extend_box(_L.box, L.box))
                    comp_N(Link, L_, Et, rng, dir)  # L_+=nodet, L.rim_t+=Link
        _L_=L_; rng+=1
    return N_, rng, Et

'''
G.DerH sums krim _G.derHs, not from links, so it's empty in the first loop.
_G.derHs can't be empty in comp_krim: init in loop link.derHs
link.DerH is ders from comp G.DerH in comp_krim
G.extH sums link.DerHs: '''

def comp_krim(link, G_, nrng, fd=0):  # sum rim _G.derHs, compare to form link.DerH layer

    _G,G = link.nodet  # same direction
    ave = G_aves[fd]
    for node in _G, G:
        if node in G_: continue  # new krim is already added
        krim = []  # kernel rim
        for _node in node.kH[-1]:
            for _link in _node.rim:
                __node = _link.nodet[0] if _link.nodet[1] is _node else _link.nodet[1]
                krim += [_G for _G in __node.kH[-1] if _G not in krim]
                if node.DerH: node.DerH.add_(__node.derH, irdnt=_node.Et[2:])
                else:         node.DerH = deepcopy(__node.derH)  # init
        node.kH += [krim]
    _xrim = list(set(_G.kH[-1]) - set(G.kH[-1]))  # exclusive kernel rim
    xrim = list(set(G.kH[-1]) - set(_G.kH[-1]))
    dderH = comp_N_(_xrim, xrim)

    if dderH.Et[0] > ave * dderH.Et[2]:
        G_ += [_G,G]  # update node_, use nested link.derH vs DerH?
        link.DerH.H[-1].add_(dderH, irdnt=dderH.H[-1].Et[2:]) if len(link.DerH.H)==nrng else link.DerH.append_(dderH,flat=1)

    # connectivity eval in segment_graph via decay = (link.relt[fd] / (link.derH.n * 6)) ** nrng  # normalized decay at current mediation

def sum_N_(N_, fd=0):  # to sum kernel layer and partial graph comp

    N = N_[0]
    n = N.n; S = N.S
    L, A = (N.distance, N.angle) if fd else (len(N.node_), N.A)
    latuple = deepcopy(N.latuple)  # ignore if Clink?
    if not fd: iderH = deepcopy(N.iderH)
    derH = deepcopy(N.derH)
    extH = deepcopy(N.extH)
    # Et = copy(N.Et)
    for N in N_[1:]:
        latuple = [P+p for P,p in zip(latuple[:-1],N.latuple[:-1])] + [[A+a for A,a in zip(latuple[-1],N.latuple[-1])]]
        n += N.n; S += N.S
        L += N.distance if fd else len(N.node_)
        A = [Angle+angle for Angle,angle in zip(A, N.angle if fd else N.A)]
        if N.iderH: iderH.add_(N.iderH)
        if N.derH: derH.add_(N.derH)
        if N.extH: extH.add_(N.extH)
        # Et = np.add(Et,N.Et)

    return n, L, S, A, derH, extH, latuple, iderH  # the last two if not fd, not sure Et is comparable

def comp_N_(_node_, node_):  # first part of comp_G to compare partial graphs for merge

    dderH = CH()
    fd = isinstance(_node_[0], Clink)
    _n, _L, _S, _A, _derH, _extH, _latuple, _iderH = sum_N_(_node_, fd)
    n, L, S, A, derH, extH, latuple, iderH = sum_N_(node_, fd)
    rn = _n/n
    et, rt, md_ = comp_ext(_L,L, _S,S/rn, _A,A)
    if fd:
        dderH.n = 1; dderH.Et = et; dderH.relt = rt
        dderH.H = [CH(Et=copy(et),relt=copy(rt),H=md_,n=1)]
    else:
        Et, Rt, Md_ = comp_latuple(_latuple, latuple, rn, fagg=1)
        dderH.n = 1; dderH.Et = np.add(Et,et); dderH.relt = np.add(Rt,rt)
        dderH.H = [CH(Et=et,relt=rt,H=md_,n=.5),CH(Et=Et,relt=Rt,H=Md_,n=1)]
        _iderH.comp_(iderH, dderH, rn, fagg=1, flat=0)

    if _derH and derH: _derH.comp_(derH, dderH, rn, fagg=1, flat=0)  # append and sum new dderH to base dderH
    if _extH and extH: _extH.comp_(extH, dderH, rn, fagg=1, flat=1)

    return dderH

def comp_N(Link, node_, iEt, rng=None, dir=None):  # rng,dir if fd, Link+=dderH, comparand rim+=Link

    fd = dir is not None  # compared links have binary relative direction?
    dderH = CH(); _N, N = Link.nodet; rn = _N.n / N.n

    if fd:  # Clink Ns
        _A, A = _N.angle, N.angle if dir else [-d for d in N.angle] # reverse angle direction for left link
        Et, rt, md_ = comp_ext(_N.distance,N.distance, _N.S,N.S/rn, _A,A)
        # Et, Rt, Md_ = comp_latuple(_G.latuple, G.latuple,rn,fagg=1)  # low-value comp in der+
        dderH.n = 1; dderH.Et = Et; dderH.relt = rt
        dderH.H = [CH(Et=copy(Et),relt=copy(rt),H=md_,n=1)]
    else:  # CG Ns
        et, rt, md_ = comp_ext(len(_N.node_),len(N.node_), _N.S,N.S/rn, _N.A,N.A)
        Et, Rt, Md_ = comp_latuple(_N.latuple, N.latuple, rn,fagg=1)
        dderH.n = 1; dderH.Et = np.add(Et,et); dderH.relt = np.add(Rt,rt)
        dderH.H = [CH(Et=et,relt=rt,H=md_,n=.5),CH(Et=Et,relt=Rt,H=Md_,n=1)]
        # / PP:
        _N.iderH.comp_(N.iderH, dderH, rn, fagg=1, flat=0)  # always >1P in compared PPs?
    # / N, if >1 PPs | Gs:
    if _N.derH and N.derH: _N.derH.comp_(N.derH, dderH, rn, fagg=1, flat=0)  # append and sum new dderH to base dderH
    if _N.extH and N.extH: _N.extH.comp_(N.extH, dderH, rn, fagg=1, flat=1)

    if fd: Link.derH.append_(dderH, flat=1)  # append dderH.H to link.derH.H
    else:  Link.derH = dderH
    iEt[:] = np.add(iEt,dderH.Et)  # init eval rng+ and form_graph_t by total m|d?
    fin = 0
    for i in 0, 1:
        Val, Rdn = dderH.Et[i::2]
        if Val > G_aves[i] * Rdn: fin = 1
        _N.Et[i] += Val; N.Et[i] += Val  # not selective
        _N.Et[2+i] += Rdn; N.Et[2+i] += Rdn  # per fork link in both Gs
        # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dderH.Et[i::2])]
    if fin:
        Link.S += (_N.S + N.S) / 2
        Link.n = min(_N.n,N.n) # comp shared layers
        for node,_node in (_N,N),(N,_N):
            node_ += [node]  # for selective kernel init or der+'rng+
            if fd:
                if len(node.rimt_) == rng:
                    node.rimt_[-1][1-dir] += [Link]  # add in last rng layer, opposite _N,N dir
                else:
                    node.rimt_ += [[[Link],[]]] if dir else [[[],[Link]]]  # add rng layer
                # add prelinks for next rng+:
                node.prerimt[1-dir] += _node.rimt[0][0] + _node.rimt[0][1]
            else:
                node.rim += [Link]

def comp_ext(_L,L,_S,S,_A,A):  # compare non-derivatives:

    dL = _L - L;      mL = min(_L,L) - ave_L  # direct match
    dS = _S/_L - S/L; mS = min(_S,S) - ave_L  # sparsity is accumulated over L
    mA, dA = comp_angle(_A,A)  # angle is not normalized
    M = mL + mS + mA
    D = abs(dL) + abs(dS) + abs(dA)  # signed dA?
    mrdn = M > D; drdn = D<= M
    mdec = mL/ max(L,_L) + mS/ max(S,_S) if S or _S else 1 + mA  # Amax = 1
    ddec = max_dist + mL/ (L+_L) + dS/ (S+_S) if S or _S else 1 + dA

    return [M,D,mrdn,drdn], [mdec,ddec], [mL,dL, mS,dS, mA,dA]


def form_graph_t(root, N_, Et, rng):  # form Gm_,Gd_ from same-root nodes
    '''
    der+: comp link via link.node_ -> dual trees of matching links in der+rng+, more likely but complex: higher-order links
    rng+: less likely: term by >d/<m, but G M projects interrupted match, and less complex?
    '''
    node_t = []
    for fd in 0,1:
        if Et[fd] > ave * Et[2+fd]:
            if not fd:  # nodes have roots
                for G in N_: G.root = []
            graph_ = segment_N_(root, N_, fd, rng)
            for graph in graph_:
                Q = graph.link_ if fd else graph.node_
                if len(Q) > ave_L and graph.Et[fd] > G_aves[fd] * graph.Et[fd]:
                    if fd: add_der_attrs(Q)
                    # else sub+rng+: comp Gs at distance < max_dist * rng+1:
                    agg_recursion(root, graph, Q, rng+1, fagg=1-fd)  # graph.node_ is not node_t yet, rng for rng+ only
                elif graph.derH:
                    root.fback_ += [graph.derH]
                    feedback(root, fd)  # update root.root.. per sub+
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]

    return node_t

'''
cluster by weights of shared links + similarity of partial clusters, initially single linkage,
similar to parallelized https://en.wikipedia.org/wiki/Watershed_(image_processing).

Then recursive sub-clustering: node assign by all in-graph weights, added in merges. 
'''
def segment_N_(root, iN_, fd, rng):

    N_ = []  # init Gts per node in iN_, merge if Lrim overlap + similarity of exclusive node_s
    max_ = []
    for N in iN_:
        # init graphts:
        rim = N.rim if isinstance(N,CG) else [L for rim_ in N.rim_t for rim in rim_ for L in rim]  # flatten rim_t
        _N_t = [[_N for L in rim for _N in L.nodet if _N is not N], [N]]  # [ext_N_, int_N_]
        # node_, link_, Lrim, Nrim_t, Et:
        Gt = [[N],[],copy(rim),_N_t,[0,0,0,0]]
        N.root = Gt
        N_ += [Gt]
        if not fd:  # def Gt seeds as local maxes, | affinity clusters | samples?:
            if not any([eN.DerH.Et[0] > N.DerH.Et[0] or (eN in max_) for eN in _N_t[0]]):  # _N if _N.V == N.V
                max_ += [N]  # V * k * max_rng: + mediation if no max in rrim?
    max_ = [N.root for N in max_]
    for Gt in N_: Gt[3][0] = [_N.root for _N in Gt[3][0]]  # replace extNs with their Gts
    # merge with connected _Gts:
    for Gt in max_ if max_ else N_:
        node_, link_, Lrim, Nrim_t, Et = Gt
        Nrim = Nrim_t[0]
        for _Gt,_link in zip(Nrim, Lrim):
            if _Gt not in N_:
                continue  # was merged
            oL_ = set(Lrim).intersection(set(_Gt[2])).union([_link])  # shared external links + potential _link
            oV = sum([L.derH.Et[fd] - ave * L.derH.Et[2+fd] for L in oL_])
            # Nrim similarity = relative V deviation,
            # + partial G similarity:
            if len(node_)/len(Nrim) > ave_L and len(_Gt[0])/len(_Gt[3][1]) > ave_L:
                sN_ = set(node_); _sN_ = set(_Gt[0])
                oN_ = sN_.intersection(_sN_)  # Nrim overlap
                xN_ = list(sN_-oN_) # exclusive node_
                _xN_ = list(_sN_-oN_)
                if _xN_ and xN_:
                    dderH = comp_N_(_xN_, xN_)
                    oV + (dderH.Et[fd] - ave * dderH.Et[2+fd])  # norm by R, * dist_coef * agg_coef?
            if oV > ave:
                Gt[1] += [_link]
                merge(Gt,_Gt); N_.remove(_Gt)

    return [sum2graph(root, Gt, fd, rng) for Gt in N_]

def merge(Gt, gt):

    N_,L_, Lrim, Nrim_t, Et = Gt
    n_,l_, lrim, nrim_t, et = gt
    N_ += n_
    L_ += l_  # internal, no overlap
    Lrim[:] = list(set(Lrim + lrim))  # exclude shared external links
    Nrim_t[:] = [[G for G in nrim_t[0] if G not in Nrim_t[0]], list(set(Nrim_t[1] + nrim_t[1]))]  # exclude shared external nodes
    Et[:] = np.add(Et,et)

def add_der_attrs(link_):
    for link in link_:
        link.extH = CH()  # for der+
        link.root = None  # dgraphs containing link
        link.rimt_ = [[[],[]]]  # dirt per rng layer
        link.prerimt = [N.rim if isinstance(N,CG) else N.rimt_[0][0] + N.rimt_[0][1] for N in link.nodet]  # next rng prelinks
        link.compared_ = []
        link.med = 1  # comp med rng, replaces len rim_
        link.dir = 0  # direction of comparison if not G0,G1, for comp link?
        link.Et = [0,0,0,0]

# not revised
def sum2graph(root, grapht, fd, rng):  # sum node and link params into graph, aggH in agg+ or player in sub+

    G_, Link_, _, _, Et = grapht
    graph = CG(fd=fd, node_=G_,link_=Link_, rng=rng, Et=Et)
    if fd:
        graph.root = root
    extH = CH()
    for G in G_:
        extH.append_(G.extH,flat=1)if graph.extH else extH.add_(G.extH)
        graph.area += G.area
        graph.box = extend_box(graph.box, G.box)
        if isinstance(G, CG):  # add latuple to Clink too?
            graph.latuple = [P+p for P,p in zip(graph.latuple[:-1],G.latuple[:-1])] + [[A+a for A,a in zip(graph.latuple[-1],G.latuple[-1])]]
            if G.iderH:  # empty in single-P PP|Gs
                graph.iderH.add_(G.iderH)
        graph.n += G.n  # non-derH accumulation?
        graph.derH.add_(G.derH)
        if fd: G.Et = [0,0,0,0]  # reset in last form_graph_t fork, Gs are shared in both forks
        else:  G.root = graph  # assigned to links if fd else to nodes?

    for link in Link_:  # sum last layer of unique current-layer links
        graph.S += link.distance
        np.add(graph.A,link.angle)
        if fd: link.root = graph
    graph.derH.append_(extH, flat=0)  # graph derH = node derHs + [summed Link_ derHs]
    if fd:
        # assign alt graphs from d graph, after both linked m and d graphs are formed
        for link in graph.link_:
            mgraph = link.root
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]
    return graph

def feedback(root, fd):  # called from form_graph_, append new der layers to root

    DerH = deepcopy(root.fback_.pop(0))  # init
    while root.fback_:
        derH = root.fback_.pop(0)
        DerH.add_(derH)
    if DerH.Et[fd] > G_aves[fd] * DerH.Et[fd+2]:
        root.derH.add_(DerH)
    if root.root and isinstance(root.root, CG):  # not Edge
        rroot = root.root
        if rroot:
            fback_ = rroot.fback_  # always node_t if feedback
            if fback_ and len(fback_) == len(rroot.node_):  # after all nodes sub+
                feedback(rroot, fd)  # sum2graph adds higher aggH, feedback adds deeper aggH layers