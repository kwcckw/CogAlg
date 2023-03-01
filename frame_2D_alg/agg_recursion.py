import sys
import numpy as np
from itertools import zip_longest
from copy import deepcopy, copy
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
import math as math
from comp_slice import *

'''
Blob edges may be represented by higher-composition patterns, etc., if top param-layer match,
in combination with spliced lower-composition patterns, etc, if only lower param-layers match.
This may form closed edge patterns around flat core blobs, which defines stable objects. 
Graphs are formed from blobs that match over <max distance. 
-
They are also assigned adjacent alt-fork graphs, which borrow predictive value from the graph. 
Primary value is match, so difference patterns don't have independent value. 
They borrow value from proximate or average match patterns, to the extent that they cancel their projected match. 
But alt match patterns borrow already borrowed value, which may be too tenuous to track, we can use the average instead.
-
Graph is abbreviated to G below:
Agg+ cross-comps top Gs and forms higher-order Gs, adding up-forking levels to their node graphs.
Sub+ re-compares nodes within Gs, adding intermediate Gs, down-forking levels to root Gs, and up-forking levels to node Gs.
-
Generic graph is a dual tree with common root: down-forking input node Gs and up-forking output graph Gs. 
This resembles neuron, which has dendritic tree as input and axonal tree as output.
But there is a radical difference: recursively structured param sets are packed in each level of these trees.
Diagram: 
https://github.com/boris-kz/CogAlg/blob/76327f74240305545ce213a6c26d30e89e226b47/frame_2D_alg/Illustrations/generic%20graph.drawio.png
'''
# aves defined for rdn+1:
aveG = 6  # fixed costs per G
aveGm = 5  # for inclusion in graph
aveGd = 4
G_aves = [aveGm, aveGd]
ave_med = 3  # call cluster_node_layer
ave_rng = 3  # rng per combined val
ave_ext = 5  # to eval comp_inder_
ave_len = 3
ave_distance = 5
ave_sparsity = 2

class Clink_(ClusterStructure):
    Q = list
    val = float
    Qm = list
    mval = float
    Qd = list
    dval = float

class CpH(ClusterStructure):  # hierarchy of params + associated vars in pplayers | players | ptuples

    H = list  # hierarchy of pplayers | players | ptuples, can be sequence
    val = float
    rdn = lambda: 1  # for all Qs?
    rng = lambda: 1
    fds = list  # m|d in pplayers,players,ptuples, m|d|None in levs?
    nval = int  # of open links: base alt rep
    # in xpplayers and derGs, each m|d:
    L = list  # der L, init None
    S = int  # sparsity: ave len link
    A = list  # area|axis: Dy,Dx, ini None


class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    G = lambda: None  # same-scope lower der|rng G.G.G.., for all nodes beyond PP
    root = lambda: None  # root graph or inder_ G, element of ex.H[-1][fd]
    # up,down trees:
    ex = object # ex inder_ ) link_) uH: Lev+= root tree slice: reforward, all forks for comp?
    inder_ = list  # inder_ ) node_) wH: Lev+= node tree slice: feedback, Lev/agg+, lev/sub+?
    # inder_ params:
    node_ = list  # single-fork, conceptually H[0], concat sub-node_s in ex.H levs
    fterm = lambda: 0  # G.node_ specification was terminated
    Link_ = lambda: Clink_()  # unique links within node_, -> inder_?
    H = list  # Lev+= node'tree'slice, ~up-forking Levs in ex.H?
    val = int
    fds = list
    rdn = lambda: 1
    rng = lambda: 1
    nval = int  # of open links: base alt rep
    x0 = float
    y0 = float  # center: box x0|y0 + L/2
    xn = float  # max distance from center
    yn = float
    L = list  # der L, init None
    S = int  # sparsity: ave len link
    A = list  # area|axis: Dy,Dx, ini None
    alt_graph_ = list  # contour + overlapping contrast graphs
    alt_Graph = None  # conditional, summed and concatenated params of alt_graph_


class CderG(ClusterStructure):  # graph links, within root node_

    node0 = lambda: Cgraph()  # converted to list in recursion
    node1 = lambda: Cgraph()
    minder_ = list  # in alt/contrast if open
    dinder_ = list
    S = int  # sparsity: ave len link
    A = list  # area and axis: Dy,Dx


def agg_recursion(root, fseg):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

    # what would be a suitable value for new level's fds here?
    root.H += [CpH(fds=[0,1], H=[Cgraph(),Cgraph()])]  # to sum feedback from new graphs
    for G in root.node_:
        G.ex.H += [CpH(fds=[0,1], H=[Cgraph(),Cgraph()])]
        # not for sub+: lower node =G.G?
    mgraph_, dgraph_ = form_graph_(root, fsub=0)  # node.H cross-comp and graph clustering, comp frng pplayers

    for fd, graph_ in enumerate([mgraph_,dgraph_]):  # eval graphs for sub+ and agg+:
        val = sum([graph.val for graph in graph_])
        # intra-graph sub+ comp node:
        if val > ave_sub * root.rdn:  # same in blob, same base cost for both forks
            # estimate rdn, assign to the weaker per sub_graphs feedback:
            for graph in graph_: graph.rdn+=1
            sub_recursion_g(graph_, fseg, root.fds + [fd])  # subdivide graph_ by der+|rng+, feedback / select graph
        # cross-graph agg+ comp graph:
        if val > G_aves[fd] * ave_agg * root.rdn and len(graph_) > ave_nsub:
            for graph in graph_: graph.rdn+=1  # estimate
            agg_recursion(root, fseg=fseg)  # replaces root.H
        else:
            root.fterm = 1
            feedback(root)  # bottom-up feedback: root.ex.H[fds].node_ = graph_, etc, breadth-first


def form_graph_(root, fsub): # form inder_ in agg+ or sub-pplayer in sub+, G is node in GG graph

    G_ = root.node_
    comp_G_(G_, fsub=fsub)  # cross-comp all graph nodes in rng, graphs may be segs | fderGs, root G += link, link.node
    # retun minder_, dinder_?
    mnode_, dnode_ = [], []  # Gs with >0 +ve fork links:
    for G in G_:
        if G.ex.node_.Qm: mnode_ += [G]  # all nodes with +ve links, not clustered in graphs yet
        if G.ex.node_.Qd: dnode_ += [G]
    graph_t = []
    for fd, node_ in enumerate([mnode_, dnode_]):
        graph_ = []  # init graphs by link val:
        while node_:  # all Gs not removed in add_node_layer
            G = node_.pop(); gnode_ = [G]
            val = add_node_layer(gnode_, node_, G, fd, val=0)  # recursive depth-first gnode_+=[_G]
            graph_ += [CpH(H=gnode_, val=val)]
        # reform graphs by node val:
        regraph_ = graph_reval(graph_, [aveG for graph in graph_], fd)  # init reval_ to start
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
            # root for feedback: sum val,node_, then selective unpack?
        graph_t += [graph_]

    # add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
    return graph_t

def graph_reval(graph_, reval_, fd):  # recursive eval nodes for regraph, increasingly selective with reduced node.link_.val

    regraph_, rreval_ = [],[]
    Reval = 0
    while graph_:
        graph = graph_.pop()
        reval = reval_.pop()
        if reval < aveG:  # same graph, skip re-evaluation:
            regraph_ += [graph]; rreval_ += [0]
            continue
        while graph.H:  # some links will be removed, graph may split into multiple regraphs, init each with graph.Q node:
            regraph = CpH()
            node = graph.H.pop()  # node_, not removed below
            val = [node.ex.node_.mval, node.ex.node_.dval][fd]  # in-graph links only
            if val > G_aves[fd]:  # else skip
                regraph.H = [node]; regraph.val = val  # init for each node, then add _nodes
                readd_node_layer(regraph, graph.H, node, fd)  # recursive depth-first regraph.Q+=[_node]
            reval = graph.val - regraph.val
            if regraph.val > aveG:
                regraph_ += [regraph]; rreval_ += [reval]; Reval += reval
    if Reval > aveG:
        regraph_ = graph_reval(regraph_, rreval_, fd)  # graph reval while min val reduction

    return regraph_

def readd_node_layer(regraph, graph_H, node, fd):  # recursive depth-first regraph.Q+=[_node]

    for link in [node.ex.node_.Qm, node.ex.node_.Qd][fd]:  # all positive
        _node = link.node1 if link.node0 is node else link.node0
        _val = [_node.ex.node_.mval, _node.ex.node_.dval][fd]
        if _val > G_aves[fd] and _node in graph_H:
            regraph.H += [_node]
            graph_H.remove(_node)
            regraph.val += _val
            readd_node_layer(regraph, graph_H, _node, fd)

def add_node_layer(gnode_, G_, G, fd, val):  # recursive depth-first gnode_+=[_G]

    for link in G.ex.node_.Q:  # all positive
        _G = link.node1 if link.node0 is G else link.node0
        if _G in G_:  # _G is not removed in prior loop
            gnode_ += [_G]
            G_.remove(_G)
            val += [_G.ex.node_.mval,_G.ex.node_.dval][fd]
            val += add_node_layer(gnode_, G_, _G, fd, val)
    return val

def comp_G_(G_, pri_G_=None, f1Q=1, fsub=0):  # cross-comp Graphs if f1Q else G_s, or segs inside PP?

    if not f1Q: minder__,dinder__ = [],[]

    for i, _iG in enumerate(G_ if f1Q else pri_G_):  # G_ is node_ of root graph
        for iG in G_[i+1:] if f1Q else G_:  # compare each G to other Gs in rng, bilateral link assign, val accum:
            # if the pair was compared in prior rng+:
            if iG in [node for link in _iG.ex.node_.Q for node in [link.node0,link.node1]]:  # if f1Q? add frng to skip?
                continue
            dx = _iG.x0 - iG.x0; dy = _iG.y0 - iG.y0  # center x0,y0
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity, proximity = ave-distance
            if distance < ave_distance * ((_iG.val + iG.val) / (2*sum(G_aves))):
                # same for cis and alt Gs:
                for _G, G in ((_iG, iG), (_iG.alt_Graph, iG.alt_Graph)):
                    if not _G or not G:  # or G.val
                        continue
                    minder_, dinder_, mval, dval, tval = comp_GQ(_G,G, fsub)  # comp inder_,LSA? comp node_? comp H?
                    if _G.S and G.S and tval > aveG:
                        mext,dext = comp_ext(_G.L,_G.S,_G.A, G.L,G.S,G.A)  # per 0G: GQ is one distributed node
                        minder_+=[mext]; dinder_+=[mext]; mval+=sum(mext); dval+=sum(dext)  # no separate rdn?
                    else:
                        minder_+=[[]]; dinder_+=[[]]
                    derG = CderG(node0=_G, node1=G, minder_=minder_, dinder_=dinder_, S=distance, A=[dy, dx])
                    # add links:
                    _G.ex.node_.Q += [derG]; _G.ex.node_.val += tval  # combined +-links val
                    G.ex.node_.Q += [derG]; G.ex.node_.val += tval
                    if mval > aveGm:
                        _G.ex.node_.Qm += [derG]; _G.ex.node_.mval += mval  # no dval for Qm
                        G.ex.node_.Qm += [derG]; G.ex.node_.mval += mval
                    if dval > aveGd:
                        _G.ex.node_.Qd += [derG]; _G.ex.node_.dval += dval  # no mval for Qd
                        G.ex.node_.Qd += [derG]; G.ex.node_.dval += dval

                    if not f1Q:  # comp G_s
                        minder__+= minder_; dinder__+= dinder_  # minder__ and dinder__ is flat, so the bracket is not needed
                # implicit cis, alt pair nesting in minder_, dinder_
    if not f1Q:
        return minder__, dinder__  # or packed in links?


def comp_GQ(_G,G,fsub):  # compare lower-derivation G.G.s, pack results in minder__,dinder__

    minder__,dinder__ = [],[]; Mval,Dval = 0,0; Mrdn,Drdn = 1,1
    Tval= aveG+1  # start
    while (_G and G) and Tval > aveG:  # same-scope if sub+, no agg+ G.G

        minder_, dinder_, mval, dval, mrdn, drdn = comp_G(_G, G, fsub, fex=0)
        minder__+=minder_; dinder__+=dinder_
        Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn  # also /rdn+1: to inder_?
        # comp ex:
        if (Mval + Dval) * _G.ex.val * G.ex.val > aveG:
            mex_, dex_, mval, dval, mrdn, drdn = comp_G(_G.ex, G.ex, fsub, fex=1)
            minder__+=mex_; dinder__+=dex_
            Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn
        else:
            minder__+=[[]]; dinder__+=[[]]
        # no ext added within GQ
        _G = _G.G
        G = G.G
        Tval = (Mval+Dval) / (Mrdn+Drdn)

    return minder__, dinder__, Mval, Dval, Tval

def comp_G(_G, G, fsub, fex):

    inder_, node_, H = G.inder_, G.node_.Q if fex else G.node_, G.H  # node_ is link_ if fex
    _inder_,_node_,_H = _G.inder_, _G.node_.Q if fex else G.node_,_G.H

    minder_, dinder_ = [],[]  # ders of implicitly nested list of pplayers in inder_
    Mval,Dval = 0,0
    Mrdn,Drdn = 1,1
    minder_,dinder_, Mval,Dval, Mrdn,Drdn = \
        comp_inder_(_inder_,inder_, minder_,dinder_, Mval,Dval, Mrdn,Drdn)
    # spec+:
    if (Mval+Dval) * _G.val*G.val * len(_node_)*len(node_) > aveG:
        if fex:  # comp link_
            sub_minder_,sub_dinder_ = comp_derG_(_node_, node_, G.fds[-1])
        else:    # comp node_
            sub_minder_,sub_dinder_ = comp_G_(_node_, node_, f1Q=0, fsub=fsub)
        Mval += sum([mxpplayers.val for mxpplayers in sub_minder_])  # add rdn?
        Dval += sum([dxpplayers.val for dxpplayers in sub_dinder_])
        # sub node_ in sub Gs
        if (Mval+Dval) * _G.val*G.val * len(_H)*len(H) > aveG:
            # comp uH|wH, wH is empty in top-down comp?
            for _forks, forks in zip(_H, H):
                for _g, _fd in zip(forks.H, forks.fds):
                    for g, fd in zip(forks.H, forks.fds):
                        if _fd == fd:
                            # below need update? compare g.inder_< sum them to minder_ and dinder_?
                            mpH, dpH = comp_pH(_g.pplayers, g.pplayers)
                            sum_pH(minder_, mpH); sum_pH(dinder_, dpH)
    else: _G.fterm=1
    # no G.fterm=1: it has it's own specification?
    # comp alts,val,rdn?
    return minder_, dinder_, Mval, Dval, Mrdn, Drdn

def comp_derG_(_derG_, derG_, fd):

    mlink_,dlink_ = [],[]
    for _derG in _derG_:
        for derG in derG_:
            mlink, dlink,_,_,_,_,= comp_inder_([_derG.minder_,_derG.dinder_][fd], [derG.dinder_,_derG.dinder_][fd], [],[],0,0,1,1)
            # add comp fds: may be different?
            mext, dext = comp_ext(1,_derG.S,_derG.A, 1,derG.S,derG.A)
            mlink_ += [mlink] + [mext]  # not sure
            dlink_ += [dlink] + [dext]

    return mlink_, dlink_


def comp_inder_(_inder_, inder_, minder_,dinder_, Mval,Dval, Mrdn,Drdn):

    nLev = 0  # lenLev = (end*2)+2: 1, 1+2, 4+2, 10+2, 22+2, 46+2.. new ext,ex per G, not inder_?
    Tval = aveG+1
    i=0; end=1
    while end <= len(_inder_) and end <= len(inder_) and Tval > aveG:
        _Lev, Lev = _inder_[i:end], inder_[i:end]  # each Lev of implicit nesting is inder_,ext,ex formed by comp_G
        for _der,der in zip(_Lev,Lev):
            if der:
                if isinstance(der,CpH):  # pplayers or ex_pplayers after ext, incr implicit nesting in m|dpplayers:
                    mpplayers, dpplayers = comp_pH(_der, der)
                    minder_ += [mpplayers]; Mval += mpplayers.val; Mrdn += mpplayers.rdn  # add rdn in form_?
                    dinder_ += [dpplayers]; Dval += dpplayers.val; Drdn += dpplayers.rdn
                else:  # list ext
                    mext, dext = comp_ext(_der[:], der[:])
                    minder_+=[mext]; dinder_+=[dext]; Mval+=sum(mext); Dval+=sum(dext)
            else:
                minder_+=[]; dinder_+=[]
        Tval = (Mval+Dval) / (Mrdn+Drdn)  # no need to loop Levs if no eval
        i = end
        end = (end*2) + 2*(not nLev)  # for 1st Lev only
        nLev += 1
    '''
    Lev1: pps: 1 pplayers  # inder_+= hLev/ comp_G: comp(inder_, ext:G.link_ coords, G.ex)-> Levs(levs., max lenlevs = lenLevs-1
    Lev2: pps; ext,ex: lenLev = 3
    Lev3: pps; pps,ext,ex; ext,ex: lenLev = 6
    Lev4: pps; pps,ext,ex; pps,pps,ext,ex,ext,ex; ext,ex: lenLev = 12 
    Lev5: pps; pps,ext,ex; pps,pps,ext,ex,ext,ex,ext,ex; pps,pps,ext,ex,pps,pps,ext,ex,ext,ex; ext,ex: lenLev = 24
    '''
    # same fds till += [fd]
    return minder_,dinder_, Mval,Dval, Mrdn,Drdn

def comp_ext(_L,_S,_A, L,S,A):

    _sparsity = _S /(_L-1); sparsity = S /(L-1)  # average distance between connected nodes, single distance if derG
    dS = _sparsity - sparsity; mS = min(_sparsity, sparsity)
    dL = _L - L; mL = min(_L, L)

    if _A and A:  # axis: dy,dx only for derG or high-aspect Gs, both val *= aspect?
        if isinstance(_A, list):
            mA, dA = comp_angle(None, _A, A)
        else:  # scalar mA or dA
            dA = _A - A; mA = min(_A, A)
    else:
        mA,dA = 0,0
    return (mL,mS,mA), (dL,dS,dA)

def comp_pH(_pH, pH):  # recursive unpack inder_s ( pplayer ( players ( ptuples -> ptuple:

    mpH, dpH = CpH(), CpH()  # new players in same top inder_?

    for i, (_spH, spH) in enumerate(zip(_pH.H, pH.H)):  # s = sub
        fd = pH.fds[i] if pH.fds else 0  # in inder_s or players
        _fd = _pH.fds[i] if _pH.fds else 0
        if _fd == fd:
            if isinstance(_spH, Cptuple):
                mtuple, dtuple = comp_ptuple(_spH, spH, fd)
                mpH.H += [mtuple]; mpH.val += mtuple.val; mpH.fds += [0]  # mpH.rdn += mtuple.rdn?
                dpH.H += [dtuple]; dpH.val += dtuple.val; dpH.fds += [1]  # dpH.rdn += dtuple.rdn

            elif isinstance(_spH, CpH):
                smpH, sdpH = comp_pH(_spH, spH)
                mpH.H += [smpH]; mpH.val += smpH.val; mpH.rdn += smpH.rdn; mpH.fds += [smpH.fds]  # or 0 | fd?
                dpH.H += [sdpH]; dpH.val += sdpH.val; dpH.rdn += sdpH.rdn; dpH.fds += [sdpH.fds]

    return mpH, dpH

def sum_G(G, g):
    # sum inder_:
    fd = g.fds[-1]
    if G.ex.H:  # G summed with prior gs
        sum_inder_(G.ex.H[0].H[fd].inder_, g.inder_)
    else:
        if fd: G.ex.H = deepcopy(g.ex.H) + [CpH(H=[Cgraph(), Cgraph(inder_=copy(g.inder_))])]  # + new lev
        else:  G.ex.H = deepcopy(g.ex.H) + [CpH(H=[Cgraph(inder_=copy(g.inder_)), Cgraph()])]
    GH=[]  # sum H:
    for i, (Lev,lev) in enumerate(zip_longest(GH, g.H, fillvalue=CpH())):
        for j, (Fork,fork, Fd,fd) in enumerate(zip_longest(Lev.H,lev.H, Lev.fds,lev.fds, fillvalue=[])):
            if fork:
                if Fork:
                    if Fd==fd:
                        sum_inder_(Fork.inder_, fork.inder_)
                    else:
                        Lev.fds.insert(j,fd); Lev[i].H.insert(j, deepcopy(fork))
                else:   Lev.fds.insert(j,fd); Lev[i].H.insert(j, deepcopy(fork))  # in G.H[i].H[fork]?
    G.H = GH  # old G.H is not summed
    G.L += g.L
    G.S += g.S
    if isinstance(g.A, list):
        if g.A:
            if G.A:
                G.A[0] += g.A[0]; G.A[1] += g.A[1]
            else: G.A = copy(g.A)
    else: G.A += g.A
    G.val += g.val
    G.rdn += g.rdn
    G.nval += g.nval
    # not sure if needed:
    G.x0 = min(G.x0, g.x0); G.y0 = min(G.y0, g.y0); G.xn = max(G.xn, g.xn); G.yn = max(G.yn, g.yn)

    for node in g.node_:
        if node not in G.node_: G.node_ += [node]
    for link in g.Link_.Q:
        if link not in G.Link_.Q: G.Link_.Q += [link]
    # alts
    for alt_graph in g.alt_graph_:
        if alt_graph not in G.alt_graph:
            G.alt_graph_ += [alt_graph]
    if g.alt_Graph:
        if G.alt_Graph:
            sum_pH(G.alt_Graph, g.alt_Graph)
        else:
            G.alt_Graph = deepcopy(g.alt_graph)


def sum_pH_(PH_, pH_, fneg=0):
    for PH, pH in zip_longest(PH_, pH_, fillvalue=[]):  # each is CpH
        if pH:
            if PH:
                for Fork, fork in zip_longest(PH.H, pH.H, fillvalue=[]):
                    if fork:
                        if Fork:
                            if fork.inder_:
                                for (Pplayers, Expplayers),(pplayers, expplayers) in zip(Fork.inder_, fork.inder_):
                                    if Pplayers:   sum_pH(Pplayers, pplayers, fneg)
                                    else:          Fork.inder_ += [[deepcopy(pplayers),[]]]
                                    if Expplayers: sum_pH(Expplayers, expplayers, fneg)
                                    else:          Fork.inder_[-1][1] = deepcopy(expplayers)
                        else: PH.H += [deepcopy(fork)]
            else:
                PH_ += [deepcopy(pH)]  # CpH

def sum_pH(PH, pH, fneg=0):  # recursive unpack inder_s ( pplayers ( players ( ptuples, no accum across fd: matched in comp_pH

    for SpH, spH, Fd, fd in zip_longest(PH.H, pH.H, PH.fds, pH.fds, fillvalue=None):  # assume same forks
        if spH:
            if SpH:
                if isinstance(spH, Cptuple):  # PH is ptuples, SpH_ is ptuple
                    sum_ptuple(SpH, spH, fneg=fneg)
                else:  # PH is players, H is ptuples
                    sum_pH(SpH, spH, fneg=fneg)
            else:
                PH.fds += [fd]
                PH.H += [deepcopy(spH)]
    PH.val += pH.val
    PH.rdn += pH.rdn
    if not PH.L: PH.L = pH.L  # PH.L is empty list by default
    else:        PH.L += pH.L
    PH.S += pH.S
    if isinstance(pH.A, list):
        if pH.A:
            if PH.A:
                PH.A[0] += pH.A[0]; PH.A[1] += pH.A[1]
            else: PH.A = copy(pH.A)
    else: PH.A += pH.A

    return PH

# pending update
def sum2graph_(graph_, fd):  # sum node and link params into graph, inder_ in agg+ or player in sub+

    Graph_ = []  # Cgraphs
    for graph in graph_:  # CpHs
        if graph.val < aveG:  # form graph if val>min only
            continue
        X0,Y0 = 0,0
        for G in graph.H:  # CpH
            X0 += G.x0; Y0 += G.y0
        L = len(graph.H); X0/=L; Y0/=L; Xn,Yn = 0,0
        # conditional ex.inder_: remove if few links?
        Graph = Cgraph(fds=copy(G.fds), ex=Cgraph(node_=Clink_(),A=[0,0]), x0=X0, xn=Xn, y0=Y0, yn=Yn)
        # form G, keep iG:
        node_,Link_= [],[]
        for iG in graph.H:
            Xn = max(Xn, (iG.x0 + iG.xn) - X0)  # box xn = x0+xn
            Yn = max(Yn, (iG.y0 + iG.yn) - Y0)
            sum_G(Graph, iG)  # sum(Graph.uH[-1][fd], iG.pplayers), higher levs += G.G.pplayers | G.uH, lower scope than iGraph
            link_ = [iG.ex.node_.Qm, iG.ex.node_.Qd][fd]
            Link_ = list(set(Link_ + link_))  # unique links in node_
            G = Cgraph(fds=copy(iG.fds), G=iG, root=Graph, ex=Cgraph(node_=Clink_(),A=[0,0]))
            # sum quasi-gradient of links in ex.inder_: redundant to Graph.inder_, if len node_?:
            for derG in link_:
                sum_inder_(G.ex.inder_, [derG.minder_, derG.dinder_][fd]) # local feedback
                G.ex.S += derG.S; G.ex.A[0]+=derG.A[0]; G.ex.A[1]+=derG.A[1]
            l=len(link_); G.ex.L=l; G.ex.S/=l
            node_ += [G]
        Graph.node_ = node_ # lower nodes = G.G..; Graph.root = iG.root
        for Link in Link_:  # sum unique links
            sum_inder_(Graph.inder_, [Link.minder_, Link.dinder_][fd])
            if Graph.inder_[-1]:  # top ext
                Graph.inder_[-1][1]+=Link.S; Graph.inder_[-1][2][0]+=Link.A[0]; Graph.inder_[-1][2][1]+=Link.A[1]
            else: Graph.inder_[-1] = [1,Link.S,Link.A]
        L = len(Link_); Graph.inder_[-1][0] = L; Graph.inder_[-1][1] /= L  # last pplayers LSA per Link_
        # inder_ LSA per node_:
        Graph.A = [Xn*2,Yn*2]; L=len(node_); Graph.L=L; Graph.S = np.hypot(Xn-X0,Yn-Y0) / L
        # if val>alt_val: rdn+=len_Q?:
        Graph.val = Graph.val + sum([lev.val for lev in Graph.H]) / max(1, sum([lev.rdn for lev in Graph.H]))

        Graph_ += [Graph]
    return Graph_


def sum_inder_(Inder_, inder_, fext=1):

    for i, (Inder, inder) in enumerate(zip_longest(Inder_, inder_, fillvalue=None)):
        if inder is not None:
            if Inder:
                if inder:  # to check if pplayers is empty bracket
                    if isinstance(inder, CpH):
                        sum_pH(Inder,inder)
                    elif fext:  # fext = 0 to exclude summing ext
                        for i in range(2): Inder[i] += inder[i]  # ext params (L，S)
                        if isinstance(Inder[2], list):  # A is [0,0]
                            Inder[2][0] += inder[2][0]; Inder[2][1] += inder[2][1]
                        else:  # A is int
                            Inder[2] += inder[2]
            elif Inder is None:  # to check if inder_ has more inders
                Inder_ += [deepcopy(inder)]
            else:
                Inder_[i] = deepcopy(inder)

# draft
def sub_recursion_g(graph_, fseg, fds, RVal=0, DVal=0):  # rng+: extend G_ per graph, der+: replace G_ with derG_

    for graph in graph_:
        node_ = graph.node_
        if graph.val > G_aves[fds[-1]] and len(node_) > ave_nsub:
            # we need to add fds to new lev here, not sure if [fds[-1]] is correct
            graph.H.append(CpH(fds=[fds[-1]],  H=[Cgraph(),Cgraph()]))  # to sum new graphs
            sub_mgraph_, sub_dgraph_ = form_graph_(graph, fsub=1)  # cross-comp and clustering
            # rng+:
            Rval = sum([sub_mgraph.val for sub_mgraph in sub_mgraph_])
            if RVal + Rval > ave_sub * graph.rdn:  # >cost of call:
                rval, dval = sub_recursion_g(sub_mgraph_, fseg=fseg, fds=fds+[0], RVal=Rval, DVal=DVal)
                RVal += rval+dval
            else:
                graph.H[-1].H[0] = []; graph.H[-1].fds.insert(0,None)
            # der+:
            Dval = sum([sub_dgraph.val for sub_dgraph in sub_dgraph_])
            if DVal + Dval > ave_sub * graph.rdn:
                rval, dval = sub_recursion_g(sub_dgraph_, fseg=fseg, fds=fds+[1], RVal=Rval, DVal=DVal)
                Dval += rval+dval
            else:
                graph.H[-1].H[1] = []; graph.H[-1].fds.insert(1,None)  # fds[1]?
            RVal += Rval
            DVal += Dval
        else:
            graph.fterm = 1  # forward is terminated
            # graph.node_'s H will be empty here
            feedback(graph)  # bottom-up feedback to update root.H, then downward reforward to update node.ex.H

    return RVal, DVal  # or SVal= RVal+DVal, separate for each fork of sub+?


def feedback(root):  # bottom-up update root.H, breadth-first

    fbV = aveG+1
    while root and fbV > aveG:
        if root.fterm:  # forward was terminated in all nodes
            root.fterm = 0  # or reuse for fforward?
            fbval, fbrdn = 0,0
            for node in root.node_:
                for i, (Lev,lev) in enumerate(zip_longest(root.H[1:], node.H, fillvalue=[])):  # root.H[1:] maps to node.H
                    if lev:
                        j = sum(fd*(2**k) for k,fd in enumerate(lev.fds[i:]))
                        if not Lev:  # init:
                            Lev=CpH(H=[Cgraph() for fork in lev.fds[i:]])
                        sum_inder_(Lev.H[j].inder_, lev.H[j].inder_)  # same fork
            for Lev in root.H:
                fbval += Lev.val; fbrdn += Lev.rdn
            fbV = fbval/max(1, fbrdn)  # right now node.H could be empty
            if root.root:
                root = root.root
            else:
                break
            
    # this should be outside of while loop? else when fbV > aveG is false, fforward will not be run too
    fforward(root)  # feedback forwarding, no cross-comp

# rough draft
def fforward(root):  # top-down update node.ex.H, breadth-first

    for node in root.node_:
        for i, (rLev,nLev) in enumerate(zip_longest(root.ex.H, node.ex.H[1:], fillvalue=[])):  # root.ex.H maps to node.ex.H[1:]
            if rLev:
                j = sum(fd*(2**k) for k,fd in enumerate(rLev.fds[i:]))
                if not nLev:  # init:
                    nLev=CpH(fds=copy(rLev.fds), H=[Cgraph() for fork in rLev.fds[i:]])
                sum_inder_(nLev.H[j].inder_, rLev.H[j].inder_)  # same fork
        ffV,ffR = 0,0
        for Lev in root.ex.H:
            ffV += Lev.val; ffR += Lev.rdn

        # converted G from PP doesn't have node_ assigned
        if node.fterm and node.node_ and ffV/ffR > aveG:
            node.fterm = 0
            fforward(node)
        # no else? (should be no)

# old:
def add_alt_graph_(graph_t):  # mgraph_, dgraph_
    '''
    Select high abs altVal overlapping graphs, to compute value borrowed from cis graph.
    This altVal is a deviation from ave borrow, which is already included in ave
    '''
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            for node in graph.inder_s.H[-1].node_:
                for derG in node.link_.Q:  # contour if link.inder_s.val < aveGm: link outside the graph
                    for G in [derG.node0, derG.node1]:  # both overlap: in-graph nodes, and contour: not in-graph nodes
                        alt_graph = G.roott[1-fd]
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, CpH):  # not proto-graph or removed
                            graph.alt_graph_ += [alt_graph]
                            alt_graph.alt_graph_ += [graph]
                            # bilateral assign
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            if graph.alt_graph_:
                graph.alt_inder_s = CpH()  # players if fsub? der+: inder_s[-1] += player, rng+: players[-1] = player?
                for alt_graph in graph.alt_graph_:
                    sum_pH(graph.alt_inder_s, alt_graph.inder_s)  # accum alt_graph_ params
                    graph.alt_rdn += len(set(graph.inder_s.H[-1].node_).intersection(alt_graph.inder_s.H[-1].node_))  # overlap
