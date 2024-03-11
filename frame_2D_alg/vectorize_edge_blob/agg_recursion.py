import numpy as np
from copy import deepcopy, copy
from itertools import combinations, zip_longest
from .comp_slice import CH, Clink, CG, CP, add_, comp_, append_, get_match
from .filters import aves, ave_mL, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd
from .slice_edge import slice_edge, comp_angle
from .comp_slice import der_recursion, comp_latuple
from utils import box2center, extend_box

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

def vectorize_root(edge):  # vectorization in 3 composition levels of xcomp, cluster:

    for P in edge.P_: P.derH = CH()  # we need to add derH into P? It's not added when P is initialized
    der_recursion(None, edge)  # vertical, lateral-overlap P cross-comp -> PP clustering

    for fd, node_ in enumerate(edge.node_):  # always node_t
        if edge.iderH and edge.iderH.Et:
            val,rdn = edge.iderH.Et[fd], edge.iderH.Et[2+fd]
            if  val * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * rdn:
                # PP -> G:
                for i, PP in enumerate(node_):
                    PP.root = None  # still no decay in internal links
                    PP.node_ = PP.P_  # revert to base node_
                # discontinuous PP rng+ cross-comp, cluster -> G_t
                agg_recursion(None, edge, node_, nrng=1, fagg=1)  # agg+, no Cgraph nodes

    return edge


def agg_recursion(rroot, root, node_, nrng=1, fagg=0):  # lenH = len(root.aggH[-1][0]), lenHH: same in agg_compress

    Et = [0,0,0,0,0,0]  # eval tuple, sum from Link_
    # agg+ der=1 xcomp of new Gs if fagg, else sub+: der+ xcomp of old Gs,
    # rng+ appends rim, link.dderH:
    nrng, node_, Et_ = rng_recursion(rroot, root, node_, list(combinations(node_,r=2)) if fagg else root.link_, Et, nrng=nrng)

    form_graph_t(root, node_, Et_, Et, nrng, fagg)  # root_fd, eval der++ and feedback per Gd, not sub-recursion in Gms

    if node_ and isinstance(node_[0], list):
        rEt = root.derH.Et if root.derH else (root.derH.Et if root.derH else [0,0,0,0,0,0])
        for fd, G_ in enumerate(node_):
            if rEt[fd] * (len(G_)-1)*root.rng > G_aves[fd] * rEt[2+fd]:
                # agg+ / node_t, vs. sub+ / node_:
                agg_recursion(rroot, root, G_, nrng=1, fagg=1)
                if rroot and fd and root.derH:  # der+ only (check not empty root.derH)
                    rroot.fback_ += [root.derH]
                    feedback(rroot)  # update root.root..


def rng_recursion(rroot, root, _node_, Q, iEt, iEt_=[], nrng=1):  # rng++/G_, der+/link_ in sub+, -> rim_H

    fd = isinstance(Q[0],Clink)
    Et = [0,0,0,0,0,0]  # for rng+
    Et_ = [[0,0,0,0,0,0] for _ in _node_]  # per compared G for rng+ and node_connect
    node_ = []  # for rng+, append inside comp_G

    if fd:  # only in 1st rng+ from der+, extend root links
        for link in Q:
            G = link.node; _G = link._node
            if _G in G.compared_: continue
            if link.dderH.Et[1] > G_aves[1] * link.dderH.Et[3]:  # eval der+
                G.compared_+=[_G]; _G.compared_+=[G]
                _idx,idx = _node_.index(_G), _node_.index(G)
                link = comp_G(link, node_, Et, Et_[_idx], Et_[idx])
                comp_rim(_node_,link,nrng)  # add matching-direction rim links for next rng+?
    else:
        for _G,G in Q:  # prelinks in rng+
            if _G in G.compared_: continue
            dy,dx = box2center(G.box)
            dist = np.hypot(dy,dx)  # distance between node centers
            _idx,idx = _node_.index(_G), _node_.index(G)
            if nrng>1: _iM,_iR, iM,iR = iEt_[_idx][0],iEt_[_idx][2], Et_[idx][0],Et_[idx][2]
            # rng+ pairwise eval:
            if nrng==1 or ((iM+_iM)/ (dist/ave_distance) > ave*(iR+_iR)):  # or directional?
                G.compared_+=[_G]; _G.compared_+=[G]
                link = comp_G([_G,G, dist, [dy,dx]], node_, Et, Et_[_idx], Et_[idx])

    if Et[0] > ave_Gm * Et[2]:
        # rng+ eval per arg cluster because comp is bilateral, 2nd test per new pair
        iEt[:] = [V+v for V,v in zip(iEt, Et)]  # Vt[i]+=v; Rt[i]+=rt[i]; Dt[i]+=d
        if node_:  # eval rng+
            Et_ = [ Et_[_node_.index(node)] for node in node_ ]  # select only Et of node with added rim    
            # we shouldn't parse the deeper node_ and et_? Because the deepest node_ is empty unless the Et eval above is false
            nrng, _, _ = rng_recursion(rroot, root, node_, list(combinations(node_,r=2)), iEt, Et_, nrng+1)

    return nrng, node_,Et_


def comp_G(link, node_, iEt, _G_Et, G_Et):  # add flat dderH to link and link to the rims of comparands

    dderH = CH(nest=0, Et=[], H=[], n=1)

    if isinstance(link, Clink):  # der+ only
        _G,G = link._node, link.node; rn = _G.n/G.n
    else:  # rng+
        _G,G, dist, [dy,dx] = link; rn = _G.n/G.n
        # / P
        Et, md_ = comp_latuple(_G.latuple, G.latuple, rn, fagg=1)
        dderH.Et = Et
        dderH.H = [CH(nest=0, Et=[*Et], H=md_)]
        # / PP, if >1 Ps:
        if _G.iderH and G.iderH:
            dH = comp_(_G.iderH, G.iderH, rn, fagg=1)  # generic dderH
            append_(dderH, dH)
        link = Clink(_node=_G, node=G, S=dist, A=[dy,dx], dderH=dderH)
    # / G, if >1 PPs| Gs, both forks:
    if _G.derH and G.derH:
        dHe = comp_(_G.derH, G.derH, rn, fagg=1)
        append_(dderH, dHe, fmerge=1)
        et, extt = comp_ext((len(_G.node_),_G.S,_G.A),(len(G.node_),G.S,G.A), rn)  # or unpack?
        dHext = CH(nest=0, Et=et, H=extt, n=0.5)
        append_(dderH, dHext)
    else:
        dderH.H += [CH(nest=0, Et=[], H=[], n=0)]
        # for fixed len layer to decode nesting, else use Cext as a terminator?
    for fd in 0,1:
        Val, Rdn, Dec = Et[fd::2]
        if Val > G_aves[fd] * Rdn:
            if not fd: node_ += [_G,G]  # for rng+
            _G_Et[fd] += Val; _G_Et[2+fd] += Rdn; _G_Et[4+fd] += Dec   # for both Gs and rng+  (This should be within the eval too?)
            G_Et[fd]  += Val; G_Et[2+fd]  += Rdn; G_Et[4+fd]  += Dec 
            iEt[fd::2] = [V+v for V,v in zip(iEt[fd::2], Et[fd::2])]  # to eval grapht in form_graph_t
            if not fd:
                for G in link.node, link._node:
                    rim_H = G.rim_H
                    if rim_H and isinstance(rim_H[0],list):  # rim is converted to rim_H in 1st sub+
                        if len(rim_H) == len(G.Rim_H): rim_H += [[]]  # no new rim layer yet
                        rim_H[-1] += [link]  # rim_H
                    else:
                        rim_H += [link]  # rim
    return link

# not updated
def comp_rim(_link_, link, nrng):  # for next rng+:

    for G in link._node, link.node:
        for _link in G.rim_H[-1]:
            _G = _link.node if _link.node in [link._node,link.node] else _link.node  # new to link
            _cy, _cx = box2center(_G.box); cy, cx = box2center(_G.box); 
            dy = _cy - cy; dx = _cx - cx
            dist = np.hypot(dy, dx)  # distance between node centers
            # or use compared?
            if 2*nrng > dist > 2*(nrng-1):  # init max comparison distance = 2
                # potentially connected G,_G: within rng, no comp in prior rng
                if comp_angle(link.A,_link.A)[0] > ave:  # link direction matches in G|_G rim_H[-1]
                    _link_+= [Clink(node=G, _node=_G, A=[dy,dx], S=dist)]  # to compare in rng+ (so no dderH here? dderH will be added in the next rng+?)


def form_graph_t(root, G_, Et_, Et, nrng, fagg=0):  # form Gm_,Gd_ from same-root nodes

    node_connect(G_, Et_)  # Graph Convolution of Correlations over init _G_
    node_t = []
    for fd in 0, 1:
        if Et[fd] > ave * Et[2+fd]:  # eVal > ave * eRdn
            graph_ = segment_node_(root, G_, fd, nrng, fagg)  # fd: node-mediated Correlation Clustering
            if fd:  # der+ only, rng++ term by high diffs, can't be extended much in sub Gs
                for graph in graph_:
                    if graph.derH.nest: M,R = [graph.derH.H[-1].Et[1], graph.derH.H[-1].Et[3]] if graph.derH else [0,0]  # from last Link_
                    else:               M,R = [graph.derH.Et[1], graph.derH.Et[3]] if graph.derH else[0,0]  # single Link_
                    if graph.link_ and M > G_aves[1] * R:
                        node_ = graph.node_
                        if isinstance(node_[0].rim_H[0], CG):  # 1st sub+, same rim nesting?
                            for node in node_: node.rim_H = [node.rim_H]  # rim -> rim_H
                        agg_recursion(root, graph, graph.node_, nrng, fagg=0)
                    elif graph.derH:
                        root.fback_ += [graph.derH]
                        feedback(root)  # update root.root.. per sub+
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]
    if any(node_t):
        G_[:] = node_t  # else keep root.node_


def node_connect(iG_, iEt_):  # node connectivity = sum surround link vals, incr.mediated: Graph Convolution of Correlations
    '''
    Aggregate direct * indirect connectivity per node from indirect links via associated nodes, in multiple cycles.
    Each cycle adds contributions of previous cycles to linked-nodes connectivity, propagated through the network.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png
    '''
    _G_, _Et_ = iG_, iEt_
    while True:
        # eval accumulated G connectivity, indirect range extension
        G_, Et_ = [], []  # next connectivity expansion, more selective by DVt,Lent = [0,0],[0,0]?
        for G, et in zip(_G_, _Et_):
            uprim = []  # >ave updates of direct links
            rim = G.rim_H[-1] if G.rim_H and isinstance(G.rim_H[0], list) else G.rim_H
            for i in 0,1:
                val,rdn,dec = et[i::2]  # connect by last layer
                ave = G_aves[i]
                for link in rim:
                    # >ave derG in fd rim
                    lval,lrdn,ldec = link.dderH.Et[i::2]; ldec /= link.dderH.n
                    _G = link._node if link.node is G else link.node
                    _et = iEt_[iG_.index(_G)]
                    _val,_rdn,_dec = _et[i::2]
                    # Vt.. for segment_node_:
                    V = ldec * (val+_val); dv = V-lval
                    R = ldec * (rdn+_rdn); dr = R-lrdn
                    D = ldec * (dec+_dec); dd = D-ldec
                    link.dderH.Et[i::2] = [V, R, D]
                    if dv > ave * dr:
                        if not G.derH: G.derH.Et = [0,0,0,0,0,0]
                        G.derH.Et[i::2] = [V+v for V,v in zip(G.derH.Et[i::2], [V, R, D])]  # add link last layer vals
                        if link not in uprim: uprim += [link]
                        # more selective eval: dVt[i] += dv; L=len(uprim); Lent[i] += L
                    if V > ave * R:
                        et[i::2] = [V+v for V, v in zip(et[i::2], [dv, dr, dd])]
            if uprim:  # prune rim for next loop
                rim[:] = uprim
                G_ += [G]; Et_ += [et]
        if G_:
            _G_ = G_; _Et_ = Et_  # exclude weakly incremented Gs from next connectivity expansion loop
        else:
            break

def segment_node_(root, root_G_, fd, nrng, fagg):  # eval rim links with summed surround vals for density-based clustering

    # graph += [node] if >ave (surround connectivity * relative value of link to any internal node)
    igraph_ = []; ave = G_aves[fd]

    for G in root_G_:  # init per node
        # get last-layer Vt,Rt,Dt, can't be empty:
        Et = copy(G.derH.H[-1].Et) if G.derH.nest else copy(G.derH.Et)
        link_ = copy(G.rim_H[-1] if G.rim_H and isinstance(G.rim_H[0],list) else G.rim_H)
        grapht = [[G],[], Et, link_]  # link_ = last rim
        G.root = grapht  # for G merge
        igraph_ += [grapht]
    _graph_ = igraph_

    while True:
        graph_ = []
        for grapht in _graph_:  # extend grapht Rim with +ve in-root links
            G_, Link_, Et, Rim = grapht
            inVal, inRdn = 0,0  # new in-graph +ve
            new_Rim = []
            for link in Rim:  # unique links
                if link.node in G_:
                    G = link.node; _G = link._node
                else:
                    G = link._node; _G = link.node
                if _G in G_: continue
                # connect by rel match of nodes * match of node Vs: surround M|Ds,
                # cval suggests how deeply inside the graph is G:
                cval = link.dderH.Et[fd] + get_match(_G.derH.Et[fd], G.derH.Et[fd])  # same coef for int and ext match?
                crdn = link.dderH.Et[2+fd] + (_G.derH.Et[2+fd] + G.derH.Et[2+fd]) / 2
                if cval > ave * crdn:  # _G and its root are effectively connected
                    # merge _root:
                    _grapht = _G.root
                    _G_,_Link_,_Et,_Rim = _grapht
                    if link not in Link_: Link_ += [link]
                    Link_[:] += [_link for _link in _Link_ if _link not in Link_]
                    for g in _G_:
                        g.root = grapht
                        if g not in G_: G_+=[g]
                    for i in 0,1:
                        Et[:] = [V+v for V,v in zip(Et, _Et)]
                        inVal += _Et[fd]; inRdn += _Et[2+fd]
                    if _grapht in igraph_:
                        igraph_.remove(_grapht)
                    new_Rim += [link for link in _Rim if link not in new_Rim+Rim+Link_]
            # for next loop:
            if len(new_Rim) * inVal > ave * inRdn:
                graph_ += [[G_,Link_, Et, new_Rim]]

        if graph_: _graph_ = graph_  # selected graph expansion
        else: break

    graph_ = []
    for grapht in igraph_:
        if grapht[2][fd] > ave * grapht[2][2+fd]:  # form Cgraphs if Val > ave* Rdn
            graph_ += [sum2graph(root, grapht[:3], fd, nrng)]

    return graph_


def sum2graph(root, grapht, fd, nrng):  # sum node and link params into graph, aggH in agg+ or player in sub+

    G_, Link_, Et = grapht
    N = 0  # der accumulation span for derH
    graph = CG(fd=fd, node_=G_,link_=Link_, rng=nrng, latuple=[0,0,0,0,0,[0,0]])
    extH = CH(nest=0, Et=[0,0,0,0,0,0], H=[])
    if fd:
        graph.root = root
    for link in Link_:  # unique current-layer links
        graph.extH = add_([],extH, link.dderH)  # irdnt from link.dderH.Et?
        graph.S += link.S; np.add(graph.A,link.A)
        link.root = graph
    N += extH.n
    # grapht int = node int+ext
    for G in G_:
        N += G.n
        graph.area += G.area
        sum_last_lay(G, fd)
        graph.box = extend_box(graph.box, G.box)
        graph.latuple = [P+p for P,p in zip(graph.latuple[:-1],graph.latuple[:-1])] + [[A+a for A,a in zip(graph.latuple[-1],graph.latuple[-1])]]
        add_([],graph.iderH, G.iderH, irdnt=[1,1])
        graph.derH = add_([],graph.derH, G.derH)

    append_(graph.derH, extH)  # dsubH | daggH
    
    ''' obsolete? (i think we still need this? When fagg, we convert nest from 1 to 2 before append extH?)
    if fagg:  
        if graph.aggH and graph.aggH.nest == 1:  # if depth == 1 (derH), convert to subH (depth =2)
            # 1st agg+, init aggH = [subHv]:
            graph.aggH = CH(nest=2, Et=[*graph.aggH.Et], H=[graph.aggH], n=N)
    '''
    if fd:  # assign alt graphs from d graph, after both linked m and d graphs are formed
        for link in graph.link_:
            mgraph = link.roott[0]
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]

    return graph


# not reviewed:
def sum_last_lay(G, fd):  # eLay += last layer of link.daggH (dsubH|ddaggH)

    eDerH = CH(nest=1, Et=[0,0,0,0,0,0], H=[])
    for link in G.rim_H[-1] if G.rim_H and isinstance(G.rim_H[0],list) else G.rim_H:
        if link.dderH:
            H = link.dderH.H
            if G.derH:
                G_depth = G.derH.nest
                if G_depth == 2 and G.ederH and G.ederH.nest == 2:
                    H = H[-1].H  # last subH
                else:
                    H = []
            # derH_/ last xcomp: len subH *= 2, maybe single dderH
            eDerH.H = [add_([], eHE, eHe, irdnt=link.dderH.Et[2:4]) for eHE, eHe in zip_longest(H[int(len(H)/2): ], eDerH.H, fillvalue=CH(nest=0, Et=[0,0,0,0,0,0],H=[]))]
            # sum all derHs of link layer=rdH into esubH[-1]

    if eDerH.H: add_([],G.ederH, eDerH)

def comp_ext(_ext, ext, rn):  # primary ext only

    _L,_S,_A = _ext; L,S,A = ext; L/=rn; S/=rn  # angle is not normalized

    mA, dA = comp_angle(_A, A)
    mS, dS = min(_S,S)-ave_mL, _S/_L - S/L  # S is summed over L, dS is not summed over dL
    mL, dL = min(_L,L)-ave_mL, _L -L
    M = mL + mS + mA
    D = abs(dL) + abs(dS) + dA
    mrdn = M > D
    drdn = D<= M
    # all comparands are positive: maxm = maxd
    Lmax = max(L,_L); Smax = max(S,_S); Amax = 1
    mdec = mL/Lmax + mS/Smax + mA/Amax
    ddec = dL/Lmax + dS/Smax + dA/Amax

    return [M,D,mrdn,drdn,mdec,ddec], [mL,dL,mS,dS,mA,dA]


def feedback(root):  # called from form_graph_, append new der layers to root

    DerH = deepcopy(root.fback_.pop(0))  # init
    while root.fback_:
        derH = root.fback_.pop(0)
        add_([],DerH, derH)
        DerH = [add_([],HE,He) for HE,He in zip_longest(DerH, derH, fillvalue=CH(nest=0, Et=[0,0,0,0,0,0],H=[]))]

    if DerH.Et[1] > G_aves[1] * DerH.Et[3]:  # compress levels?
        root.derH = [add_([],rHe,He) for rHe,He in zip_longest(root.derH, DerH, fillvalue=CH(nest=0, Et=[0,0,0,0,0,0],H=[]))]
        # root.Et = [V+v for V,v in zip_longest(root.Et, ET, fillvalue=0)]

    if root.root and isinstance(root.root, CG):  # not Edge
        rroot = root.root
        if rroot:
            fback_ = rroot.fback_  # always node_t if feedback
            if fback_ and len(fback_) == len(rroot.node_[1]):  # after all nodes sub+
                feedback(rroot)    # sum2graph adds higher aggH, feedback adds deeper aggH layers