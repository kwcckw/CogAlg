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
But we have recursively structured param sets packed in each level of these trees, there is no such structure in neurons.
Diagram: 
https://github.com/boris-kz/CogAlg/blob/76327f74240305545ce213a6c26d30e89e226b47/frame_2D_alg/Illustrations/generic%20graph.drawio.png
-
Clustering criterion is G.M|D, summed across >ave vars if selective comp (<ave vars are not compared, so they don't add costs).
Fork selection should be per var or co-derived der layer or agg level. 
There are concepts that include same matching vars: size, density, color, stability, etc, but in different combinations.
Weak value vars are combined into higher var, so derivation fork can be selected on different levels of param composition.
'''
# aves defined for rdn+1:
aveG = 6  # fixed costs per G
aveGm = 5  # for inclusion in graph
aveGd = 4
G_aves = [aveGm, aveGd]
ave_med = 3  # call cluster_node_layer
ave_rng = 3  # rng per combined val
ave_ext = 5  # to eval comp_derH
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
    '''
    L = list  # der L, init None
    S = int  # sparsity: ave len link
    A = list  # area|axis: Dy,Dx, ini None
    '''

class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    G = lambda: None  # same-scope lower-der|rng G.G.G., or [G0,G1] in derG, None in PP
    root = lambda: None  # root graph or derH G, element of ex.H[-1][fd]
    aggH = lambda: list  # list of CpQ derHs: derH) node_) H: Lev+= node tree slice/fb, Lev/agg+, lev/sub+?
    valt = lambda: [0,0]
    rdnt = lambda: [1,1]
    fds = list  # or fd, with sub fds in derH?
    rng = lambda: 1
    box = lambda: [0,0,0,0,0,0]  # y,x, y0,yn, x0,xn
    node_ = list  # single-fork, conceptually H[0], concat sub-node_s in ex.H levs
    link_ = lambda: Clink_()  # temporary holder for der+ node_, then unique links within graph?
    fterm = lambda: 0  # G.node_ sub-comp was terminated
    # uH: up-forking Levs if mult roots, not implemented yet
    H = list  # down-forking tree of Levs: slice of nodes
    nval = int  # of open links: base alt rep
    alt_graph_ = list  # contour + overlapping contrast graphs
    alt_Graph = None  # conditional, summed and concatenated params of alt_graph_


def agg_recursion(root, fseg):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

    mgraph_, dgraph_ = form_graph_(root, fsub=0)  # node.H cross-comp and graph clustering, comp frng pplayers

    for fd, graph_ in enumerate([mgraph_,dgraph_]):  # eval graphs for sub+ and agg+:
        val = sum([graph.valt[fd] for graph in graph_])
        # intra-graph sub+ comp node:
        if val > ave_sub * root.rdn:  # same in blob, same base cost for both forks
            for graph in graph_: graph.rdn+=1  # estimate, assign to the weaker in feedback
            sub_recursion_g(graph_, fseg, root.fds + [fd])  # divide graph_ in der+|rng+ sub_graphs
        else:
            root.fterm = 1; feedback(root)  # update root.root..H, breadth-first
        # cross-graph agg+ comp graph:
        if val > G_aves[fd] * ave_agg * root.rdn and len(graph_) > ave_nsub:
            for graph in graph_: graph.rdn+=1  # estimate
            agg_recursion(root, fseg=fseg)  # replaces root.H
        else:
            root.fterm = 1; feedback(root)  # update root.root..H, breadth-first


def form_graph_(root, fsub): # form derH in agg+ or sub-pplayer in sub+, G is node in GG graph

    G_ = root.node_
    comp_G_(G_, fsub=fsub)  # cross-comp all graph nodes in rng, graphs may be segs | fderGs, root G += link, link.node

    mnode_, dnode_ = [], []  # Gs with >0 +ve fork links:
    for G in G_:
        if G.link_.Qm: mnode_ += [G]  # all nodes with +ve links, not clustered in graphs yet
        if G.link_.Qd: dnode_ += [G]
    graph_t = []
    for fd, node_ in enumerate([mnode_, dnode_]):
        graph_ = []  # init graphs by link val:
        while node_:  # all Gs not removed in add_node_layer
            G = node_.pop(); gnode_ = [G]
            val = add_node_layer(gnode_, node_, G, fd, val=0)  # recursive depth-first gnode_+=[_G]
            graph_ += [CpQ(H=gnode_, val=val)]
        # reform graphs by node val:
        regraph_ = graph_reval(graph_, [aveG for graph in graph_], fd)  # init reval_ to start
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd, fsub)  # sum proto-graph node_ params in graph
        graph_t += [graph_]

    # add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
    return graph_t

def graph_reval(graph_, reval_, fd):  # recursive eval nodes for regraph, after pruning weakly connected nodes
    '''
    extend with comp_centroid to adjust all links, so centrally similar nodes are less pruned?
    or centroid match is case-specific, else scale links by combined valt of connected nodes' links, in their aggQ
    '''
    regraph_, rreval_ = [],[]
    Reval = 0
    while graph_:
        graph = graph_.pop()
        reval = reval_.pop()  # each link *= other_G.aggQ.valt
        if reval < aveG:  # same graph, skip re-evaluation:
            regraph_ += [graph]; rreval_ += [0]
            continue
        while graph.H:  # links may be revalued and removed, splitting graph to regraphs, init each with graph.Q node:
            regraph = CpQ()
            node = graph.H.pop()  # node_, not removed below
            val = [node.link_.mval, node.link_.dval][fd]  # in-graph links only
            if val > G_aves[fd]:  # else skip
                regraph.H = [node]; regraph.valt[fd] = val  # init for each node, then add _nodes
                prune_node_layer(regraph, graph.H, node, fd)  # recursive depth-first regraph.Q+=[_node]
            reval = graph.valt[fd] - regraph.valt[fd]
            if regraph.valt[fd] > aveG:
                regraph_ += [regraph]; rreval_ += [reval]; Reval += reval
    if Reval > aveG:
        regraph_ = graph_reval(regraph_, rreval_, fd)  # graph reval while min val reduction

    return regraph_

def prune_node_layer(regraph, graph_H, node, fd):  # recursive depth-first regraph.Q+=[_node]

    for link in [node.link_.Qm, node.link_.Qd][fd]:  # all positive

        _node = link.G[1] if link.G[0] is node else link.G[0]
        _val = [_node.link_.mval, _node.link_.dval][fd]
        link.val += _val/len(_node.link_) - link.val  # *1/n: interaction decay with mediation order?
        '''
        adjust link val by node val, representing mediated interactions
        or node pruning doesn't change connected-node's val, that should be from all links?
        so it only prunes further mediated evaluation?
        '''
        if _val > G_aves[fd] and _node in graph_H:
            regraph.H += [_node]
            graph_H.remove(_node)
            regraph.valt[fd] += _val
            prune_node_layer(regraph, graph_H, _node, fd)

def add_node_layer(gnode_, G_, G, fd, val):  # recursive depth-first gnode_+=[_G]

    for link in G.link_.Q:  # all positive
        _G = link.G[1] if link.G[0] is G else link.G[0]
        if _G in G_:  # _G is not removed in prior loop
            gnode_ += [_G]
            G_.remove(_G)
            val += [_G.link_.mval,_G.link_.dval][fd]
            val += add_node_layer(gnode_, G_, _G, fd, val)
    return val


def comp_G_(G_, pri_G_=None, f1Q=1, fsub=0):  # cross-comp Graphs if f1Q, else G_s in comp_node_, or segs inside PP?

    if not f1Q: dderH_ = []

    for i, _iG in enumerate(G_ if f1Q else pri_G_):  # G_ is node_ of root graph
        for iG in G_[i+1:] if f1Q else G_:  # compare each G to other Gs in rng, bilateral link assign, val accum:
            # if the pair was compared in prior rng+:
            if iG in [node for link in _iG.link_.Q for node in link.node_]:  # if f1Q? add frng to skip?
                continue
            dy = _iG.box[0]-iG.box[0]; dx = _iG.box[1]-iG.box[1]  # between center x0,y0
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity, proximity = ave-distance
            if distance < ave_distance * ((sum(_iG.valt) + sum(iG.valt)) / (2*sum(G_aves))):
                # same for cis and alt Gs:
                for _G, G in ((_iG, iG), (_iG.alt_Graph, iG.alt_Graph)):
                    if not _G or not G:  # or G.val
                        continue
                    # not revised:
                    dderH, mval, dval, tval = comp_GQ(_G,G)  # comp_G while G.G, H/0G: GQ is one distributed node?
                    ext = [1,distance,[dy,dx]]  # ext -> ext pair: new der_nodes / Graph:
                    derG = Cgraph(valt=[mval,dval], G=[_G,G], derH= dderH+[[ext]], box=[])  # box is redundant to G
                    # add links:
                    _G.link_.Q += [derG]; _G.link_.val += tval  # combined +-links val
                    G.link_.Q += [derG]; G.link_.val += tval
                    if mval > aveGm:
                        _G.link_.Qm += [derG]; _G.link_.mval += mval  # no dval for Qm
                        G.link_.Qm += [derG]; G.link_.mval += mval
                    if dval > aveGd:
                        _G.link_.Qd += [derG]; _G.link_.dval += dval  # no mval for Qd
                        G.link_.Qd += [derG]; G.link_.dval += dval

                    if not f1Q: dderH_+= dderH  # comp G_s
                # implicit cis, alt pair nesting in mderH, dderH
    if not f1Q:
        return dderH_  # else no return, packed in links


def comp_GQ(_G, G):  # compare lower-derivation G.G.s, pack results in mderH_,dderH_

    dderH_ = []; Mval,Dval = 0,0; Mrdn,Drdn = 1,1; Tval= aveG+1

    while (_G and G) and Tval > aveG:  # same-scope if sub+, no agg+ G.G
        dderH, mval, dval, mrdn, drdn = comp_G(_G, G)
        dderH_+=dderH; Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn  # rdn+=1: to derH?
        _G = _G.G
        G = G.G
        Tval = (Mval+Dval) / (Mrdn+Drdn)

    return dderH_, Mval, Dval, Tval  # ext added in comp_G_, not within GQ

def comp_G(_G, G):  # in GQ

    Mval, Dval = 0,0
    Mrdn, Drdn = 1,1
    if _G.box: _derH, derH = _G.derH, G.derH
    else:
        _fd = _G.root.fds[-1] if _G.root.fds else 0; fd = G.root.fds[-1] if G.root.fds else 0
        _derH, derH = _G.derH[_fd], G.derH[fd]  # derG in comp node_?

    dderH, Mval,Dval, Mrdn,Drdn = comp_derH(_derH, derH, Mval,Dval, Mrdn,Drdn)
    # spec:
    _node_, node_ = _G.node_, G.node_  # link_ if fd, sub_node should be empty
    if (Mval+Dval)* sum(_G.valt)*sum(G.valt) * len(_node_)*len(node_) > aveG:  # / rdn?

        sub_dderH, mval, dval, mrdn, drdn = comp_G_(_node_, node_, f1Q=0)
        Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn
        # pack m|dnode_ in m|dderH: implicit?

    else: _G.fterm=1  # no G.fterm=1: it has it's own specification?
    '''
    comp alts,val,rdn?
    comp H in comp_G_?
    select >ave m|d vars only: addressable salient mset / dset in derH? 
    cluster per var or set if recurring across root: type eval if root M|D?
    '''
    return dderH, Mval,Dval, Mrdn,Drdn

'''
aggH ( subH ( derH have fixed syntax: all lower composition orders, but represented selectively,
so each element is [gap, par], where gap is n missing pars before current par.
'''
# draft:
def comp_aggH(_aggH, aggH):  # unpack aggH ( subH ( derH:

    daggH = []; valt = [0,0]; rdnt = [1,1]; elev=0

    for i, (_subH, subH) in enumerate(zip(_aggH.Q, aggH.Q)):
        if _aggH.fds[elev]!=aggH.fds[elev]:
            break
        if elev in (0,1) or not (i+1)%(2**elev):  # first 2 levs are single-element, higher levs are 2**elev elements
            elev+=1  # elevation
        dsubH = []; valt = [0,0]; rdnt = [1,1]; elay=0
        for j, (_derH, derH) in enumerate(zip(_subH.Q, subH.Q)):
            if _subH.fds[elay] != subH.fds[elay]:
                break
            if elay in (0,1) or not (j+1)%(2**elay):  # first 2 levs are single-element, higher levs are 2**elev elements
                elay+=1  # elevation
            dsubH += [comp_derH(_derH, derH, j,i)]  # comp_par(derH[0]) if j else comp_angle(derH[0]), return CpQ dderH
        daggH += [CpQ(Q=dsubH)]
        # add summing valt, rdnt

    return CpQ(Q=daggH)


def comp_derH(_derH, derH, j,k):

    dderH = CpQ; elev=0
    for i, (_ptuple,ptuple) in enumerate(zip(_derH.Q, derH.Q)):

        if _derH.fds[elev]!=derH.fds[elev]:
            break
        if elev in (0,1) or not (i+1)%(2**elev):  # first 2 levs are single-element, higher levs are 2**elev elements
            elev += 1
        if j:
            # we need local versions of comp_vertuple and comp_ptuple, in the same fashion as updated comp_ext
            if i: dtuple = comp_vertuple(_ptuple, ptuple, dderH)
            else:
                if j: dtuple = comp_ptuple(_ptuple, ptuple, dderH)
                else: dtuple = comp_ext(_ptuple, ptuple, dderH, k)  # comp_angle if k: 1st layer in lev
        dderH.Q += [dtuple]

    return dderH


def comp_ext(_ext, ext, dderH, k):
    _L,_S,_A = _ext; L,S,A = ext
    # comp ds only:
    dS = _S[1] - S[1]; mS = min(_S[1], S[1])  # average distance between connected nodes, single distance if derG
    dL = _L[1] - L[1]; mL = min(_L[1], L[1])
    if _A and A:
        # axis: dy,dx only for derG or high-aspect Gs, both val *= aspect?
        if k: dA = _A[1] - A[1]; mA = min(_A[1], A[1])  # scalar mA,dA
        else: mA, dA = comp_angle(_A, A)
    else:
        mA,dA = 0,0
    dderH.valt[0] += mL+mS+mA
    dderH.valt[1] += dL+dS+dA
    # no rdn?


def sum2graph_(graph_, fd, fsub=0):  # sum node and link params into graph, derH in agg+ or player in sub+

    Graph_ = []  # Cgraphs
    for graph in graph_:  # CpQs

        if graph.valt[fd] < aveG:  # form graph if val>min only
            continue
        Graph = Cgraph(fds=copy(graph.H[0].fds)+[fd])  # incr der
        ''' if mult roots: 
        sum_derH(Graph.uH[0][fd].derH,root.derH) or sum_G(Graph.uH[0][fd],root)? init if empty
        sum_H(Graph.uH[1:], root.uH)  # root of Graph, init if empty
        '''
        node_,Link_ = [],[]  # form G, keep iG:
        for iG in graph.H:
            sum_G(Graph, iG, fmerge=0)  # local subset of lower Gs in new graph
            link_ = [iG.link_.Qm, iG.link_.Qd][fd]  # mlink_,dlink_
            Link_ = list(set(Link_ + link_))  # unique links in node_
            G = Cgraph(fds=copy(iG.fds)+[fd], root=Graph, node_=link_, box=copy(iG.box))  # no sub_nodes in derG, remove if <ave?
            for derG in link_:
                sum_box(G.box, derG.G[0].box if derG.G[1] is iG else derG.G[1].box)
                sum_derH(G.derH, derG.derH[fd])  # two-fork derGs are not modified
                Graph.valt[0] += derG.valt[0]; Graph.valt[1] += derG.valt[1]
            add_ext(G.box, len(link_), G.derH[-1])  # composed node ext, not in derG.derH
            # if mult roots: sum_H(G.uH[1:], Graph.uH)
            node_ += [G]
        Graph.root = iG.root  # same root, lower derivation is higher composition
        Graph.node_ = node_  # G| G.G| G.G.G..
        for derG in Link_:  # sum unique links, not box
            sum_derH(Graph.derH, derG.derH[fd])
            Graph.valt[0] += derG.valt[0]; Graph.valt[1] += derG.valt[1]
        Ext = [0,0,[0,0]]
        Ext = [sum_ext(Ext, G.derH[-1][1]) for G in node_]  # add composed node Ext
        add_ext(Graph.box, len(node_), Ext) # add composed graph Ext
        Graph.derH += [Ext]  # [node Ext, graph Ext]
        # if Graph.uH: Graph.val += sum([lev.val for lev in Graph.uH]) / sum([lev.rdn for lev in Graph.uH])  # if val>alt_val: rdn+=len_Q?
        Graph_ += [Graph]

    return Graph_

def sum_G(G, g, fmerge=0):  # g is a node in G.node_

    sum_derH(G.derH, g.derH)
    # if g.uH: sum_H(G.uH, g.uH[1:])  # sum g->G
    # if g.H: sum_H(G.H[1:], g.H)  # not used yet
    G.valt[0]+=g.valt[0]; G.valt[1]+=g.valt[1]; G.rdn += g.rdn; G.nval += g.nval
    sum_box(G.box, g.box)
    if fmerge:
        for node in g.node_:
            if node not in G.node_: G.node_ += [node]
        for link in g.Link_.Q:  # redundant?
            if link not in G.Link_.Q: G.Link_.Q += [link]
        for alt_graph in g.alt_graph_:
            if alt_graph not in G.alt_graph: G.alt_graph_ += [alt_graph]
        if g.alt_Graph:
            if G.alt_Graph: sum_G(G.alt_Graph, g.alt_Graph)
            else:           G.alt_Graph = deepcopy(g.alt_graph)
    else: G.node_ += [g]

# not fully updated
def sum_aggH(AggH, aggH):

    for SubH, subH in zip_longest(AggH.Q, aggH.Q, fillvalue=None):
        if subH:
            if SubH:
                for DerH, derH in(zip_longest(SubH.Q, subH.Q, fillvalue=None)):  # derH could be ext here? If yes we need to check and add 
                    if derH:
                        if DerH:
                            sum_derH(DerH, derH)  # probably need to extend sum_derH for ext?
                        else:
                            SubH.Q += [deepcopy(derH)] 
            else:
                AggH.Q += [deepcopy(subH)]


def add_ext(box, L, extt):  # add ext per composition level
    y,x, y0,yn, x0,xn = box
    dY = yn-y0; dX = xn-x0
    box[:2] = y/L, x/L  # norm to ave
    extt += [[L, L/ dY*dX, [dY,dX]]]  # composed L,S,A, norm S = nodes per area

def sum_ext(Ext, ext):
    for i, param in enumerate(ext):
        if i<2:  # L,S
            Ext[i] += param
        else:  # angle
            if isinstance(Ext[i], list):
                Ext[i][0] += param[0]; Ext[i][1] += param[1]
            else:
                Ext[i] += param

def sum_box(Box, box):
    Y, X, Y0, Yn, X0, Xn = Box;  y, x, y0, yn, x0, xn = box
    Box[:] = [Y + y, X + x, min(X0, x0), max(Xn, xn), min(Y0, y0), max(Yn, yn)]

def sum_H(H, h):  # add g.H to G.H, no eval but possible remove if weak?

    for i, (Lev, lev) in enumerate(zip_longest(H, h, fillvalue=[])):  # root.ex.H maps to node.ex.H[1:]
        if lev:
            if not Lev:  # init:
                Lev = CpQ(H=[[] for fork in range(2**(i+1))])
            for j, (Fork, fork) in enumerate(zip(Lev.H, lev.H)):
                if fork:
                    if not Fork: Lev.H[j] = Fork = Cgraph()
                    sum_G(Fork, fork)

# old:
def comp_pH(_pH, pH):  # recursive unpack derHs ( pplayer ( players ( ptuples -> ptuple:

    mpH, dpH = CpQ(), CpQ()  # new players in same top derH?

    for i, (_spH, spH) in enumerate(zip(_pH.H, pH.H)):  # s = sub
        fd = pH.fds[i] if pH.fds else 0  # in derHs or players
        _fd = _pH.fds[i] if _pH.fds else 0
        if _fd == fd:
            if isinstance(_spH, Cptuple):
                mtuple, dtuple = comp_ptuple(_spH, spH, fd)
                # not sure here, one of the val is always 0?
                mpH.H += [mtuple]; mpH.valt[0] += mtuple.val; mpH.fds += [0]  # mpH.rdn += mtuple.rdn?
                dpH.H += [dtuple]; dpH.valt[1] += dtuple.val; dpH.fds += [1]  # dpH.rdn += dtuple.rdn

            elif isinstance(_spH, CpQ):
                smpH, sdpH = comp_pH(_spH, spH)
                mpH.H +=[smpH]; mpH.valt[0]+=smpH.valt[0]; mpH.valt[1]+=smpH.valt[1]; mpH.rdn+=smpH.rdn; mpH.fds +=[smpH.fds]  # or 0 | fd?
                dpH.H +=[sdpH]; dpH.valt[0]+=sdpH.valt[0]; dpH.valt[1]+=sdpH.valt[1]; dpH.rdn+=sdpH.rdn; dpH.fds +=[sdpH.fds]

    return mpH, dpH

def sum_pH_(PH_, pH_, fneg=0):
    for PH, pH in zip_longest(PH_, pH_, fillvalue=[]):  # each is CpQ
        if pH:
            if PH:
                for Fork, fork in zip_longest(PH.H, pH.H, fillvalue=[]):
                    if fork:
                        if Fork:
                            if fork.derH:
                                for (Pplayers, Expplayers),(pplayers, expplayers) in zip(Fork.derH, fork.derH):
                                    if Pplayers:   sum_pH(Pplayers, pplayers, fneg)
                                    else:          Fork.derH += [[deepcopy(pplayers),[]]]
                                    if Expplayers: sum_pH(Expplayers, expplayers, fneg)
                                    else:          Fork.derH[-1][1] = deepcopy(expplayers)
                        else: PH.H += [deepcopy(fork)]
            else:
                PH_ += [deepcopy(pH)]  # CpQ

def sum_pH(PH, pH, fneg=0):  # recursive unpack derHs ( pplayers ( players ( ptuples, no accum across fd: matched in comp_pH

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
    PH.valt[0] += pH.valt[0]; PH.valt[1] += pH.valt[1]
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

# draft
def sub_recursion_g(graph_, fseg, fds, RVal=0, DVal=0):  # rng+: extend G_ per graph, der+: replace G_ with derG_

    for graph in graph_:
        node_ = graph.node_
        if graph.valt[0] > G_aves[0] and graph.valt[1] > G_aves[1] and len(node_) > ave_nsub:

            sub_mgraph_, sub_dgraph_ = form_graph_(graph, fsub=1)  # cross-comp and clustering
            # rng+:
            Rval = sum([sum(sub_mgraph.valt) for sub_mgraph in sub_mgraph_])
            # eval if Val>cost of call, else feedback per sub_mgraph?
            if RVal + Rval > ave_sub * graph.rdn:
                rval, dval = sub_recursion_g(sub_mgraph_, fseg=fseg, fds=fds+[0], RVal=Rval, DVal=DVal)
                RVal += rval+dval
            # der+:
            Dval = sum([sum(sub_dgraph.valt) for sub_dgraph in sub_dgraph_])
            if DVal + Dval > ave_sub * graph.rdn:
                rval, dval = sub_recursion_g(sub_dgraph_, fseg=fseg, fds=fds+[1], RVal=Rval, DVal=DVal)
                Dval += rval+dval
            RVal += Rval
            DVal += Dval
        else:
            graph.fterm = 1  # forward is terminated, graph.node_ is empty or weak
            feedback(graph)  # bottom-up feedback to update root.H

    return RVal, DVal  # or SVal= RVal+DVal, separate for each fork of sub+?

def feedback(root):  # bottom-up update root.H, breadth-first

    fbV = aveG+1
    while root and fbV > aveG:
        if all([[node.fterm for node in root.node_]]):  # forward was terminated in all nodes
            root.fterm = 1
            fbval, fbrdn = 0,0
            for node in root.node_:
                # node.node_ may empty when node is converted graph
                if node.node_ and not node.node_[0].box:  # link_ feedback is redundant, params are already in node.derH
                    continue
                for sub_node in node.node_:
                    fd = sub_node.fds[-1] if sub_node.fds else 0
                    if not root.H: root.H = [CpQ(H=[[],[]])]  # append bottom-up
                    if not root.H[0].H[fd]: root.H[0].H[fd] = Cgraph()
                    # sum nodes in root, sub_nodes in root.H:
                    sum_derH(root.H[0].H[fd].derH, sub_node.derH)
                    sum_H(root.H[1:], sub_node.H)  # sum_G(sub_node.H forks)?
            for Lev in root.H:
                fbval += Lev.val; fbrdn += Lev.rdn
            fbV = fbval/max(1, fbrdn)
            root = root.root
        else:
            break  # we need to break here to break from while loop above
# old:
def add_alt_graph_(graph_t):  # mgraph_, dgraph_
    '''
    Select high abs altVal overlapping graphs, to compute value borrowed from cis graph.
    This altVal is a deviation from ave borrow, which is already included in ave
    '''
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            for node in graph.derHs.H[-1].node_:
                for derG in node.link_.Q:  # contour if link.derHs.val < aveGm: link outside the graph
                    for G in [derG.node0, derG.node1]:  # both overlap: in-graph nodes, and contour: not in-graph nodes
                        alt_graph = G.roott[1-fd]
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, CpQ):  # not proto-graph or removed
                            graph.alt_graph_ += [alt_graph]
                            alt_graph.alt_graph_ += [graph]
                            # bilateral assign
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            if graph.alt_graph_:
                graph.alt_derHs = CpQ()  # players if fsub? der+: derHs[-1] += player, rng+: players[-1] = player?
                for alt_graph in graph.alt_graph_:
                    sum_pH(graph.alt_derHs, alt_graph.derHs)  # accum alt_graph_ params
                    graph.alt_rdn += len(set(graph.derHs.H[-1].node_).intersection(alt_graph.derHs.H[-1].node_))  # overlap
