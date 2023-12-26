import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from collections import deque, defaultdict
from .classes import Cgraph, CderG, CPP
from .filters import ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd, ave_dI
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_ptuple, comp_derH, sum_derH, sum_dertuple, get_match
from .agg_recursion import comp_G, comp_aggHv, comp_derHv, sum_derHv, sum_ext, sum_subHv, sum_aggHv

'''
Implement sparse param tree in aggH: new graphs represent only high m|d params + their root params.
Compare layers in parallel by flipping H, comp forks independently. Flip HH,HHH.. in deeper processing? 

1st: cluster root params within param set, by match per param between previously cross-compared nodes
2nd: cluster root nodes per above-average param cluster, formed in 1st step. 

specifically, x-param comp-> p-clustering of root AggH( SubH( DerH( Vars, after all nodes cross comp, @xp rng 
param xcomp in derivation order, nested for lower level of hierarchical xcomp?  

compress aggH: param cluster nesting reflects root param set nesting (which is a superset of param clusters). 
exclusively deeper param cluster has empty (unpacked) higher param nesting levels.

Mixed-forks: connectivity cluster must be contiguous, not uniform, as distant nodes don't need to be similar?
Nodes are connected by m|d of different param sets in links, potentially clustered in pPs for compression.

Then combine graph with alt_graphs?
'''

def root(blob, verbose):  # vectorization pipeline is 3 composition levels of cross-comp,clustering
    edge = vectorize_root(blob, verbose)
    # temporary
    for fd, G_ in enumerate(edge.node_[-1]):
        if edge.aggH:
            agg_recursion_compress(None, edge, G_, fd)


def vectorize_root(blob, verbose):  # vectorization in 3 composition levels of xcomp, cluster:

    edge, adj_Pt_ = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering

    comp_P_(edge, adj_Pt_)  # vertical, lateral-overlap P cross-comp -> PP clustering
    
    edge.node_ = [edge.node_]

    for fd, node_ in enumerate(edge.node_[-1]):  # always node_t
        if edge.valt[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.rdnt[fd]:
            for PP in node_: PP.roott = [None, None]
            agg_recursion(None, edge, node_, lenH=0, fd=0)
            # PP cross-comp -> discontinuous clustering, agg+ only, no Cgraph nodes
    return edge


def agg_recursion(rroot, root, G_, lenH, fd, nrng=1):  # compositional agg|sub recursion in root graph, cluster G_

    Et = [[0,0],[0,0],[0,0]]  # grapht link_' eValt, eRdnt, eDect(currently not used)

    if fd:  # der+
        for link in root.link_:  # reform links
            if link.Vt[1] < G_aves[1]*link.Rt[1]: continue  # maybe weak after rdn incr?
            comp_G(link._G,link.G,link, Et, lenH)
    else:   # rng+
        for i, _G in enumerate(G_):  # form new link_ from original node_
            for G in G_[i+1:]:
                dy = _G.box.cy - G.box.cy; dx = _G.box.cx - G.box.cx
                if np.hypot(dy,dx) < 2 * nrng:  # max distance between node centers, init=2
                    link = CderG(_G=_G, G=G)
                    comp_G(_G, G, link, Et, lenH)

    GG_t, valt, rdnt = form_graph_t(root, G_, Et, nrng)  # root_fd, eval sub+, feedback per graph
    
    # not sure but we can have sub+ per GG here?
    for sfd, GG_ in enumerate(GG_t):
        for GG in GG_:
            sub_recursion(root, GG, GG.node_[-1], lenH=len(GG.aggH[-1][0]), fd=sfd, nrng=1)

    if isinstance(G_[0],list):  # node_t was formed above

        for i, node_ in enumerate(G_):
            if root.valt[i] * (len(node_)-1)*root.rng > G_aves[i] * root.rdnt[i]:
                # agg+/ node_( sub)agg+/ node, vs sub+ only in comp_slice
                agg_recursion(rroot, root, node_, lenH=0, fd=0)  # der+ if fd, else rng+ =2
                if rroot:
                    rroot.fback_t[i] += [[root.aggH,root.valt,root.rdnt,root.dect]]
                    feedback(rroot,i)  # update root.root..



def agg_recursion_compress(rroot, root, G_, fd):  # compositional agg+|sub+ recursion in root graph, clustering G_

    parHv = [root.aggH,root.valt[fd],root.rdnt[fd],root.dect[fd]]
    form_pP_(pP_=[], parHv=parHv, fd=fd)  # sum is not needed here
    # compress aggH -> pP_,V,R,Y: select G' V,R,Y?

# to create 1st compressed layer
def init_parHv(parH, V, R, Y, fd):

    # 1st layer
    pP, pV,pR, pY = parH[0], parH[0][1][fd], parH[0][2][fd], parH[0][3][fd]
    pP_, part_ = [], []

    # compress 1st layer - always single element
    _,_,_, rV, rR, rY = compress_play_(pP_, [pP], part_, (0, 0, 0), (pV, pR, pY), fd)
    pP_ = [[part_,rV,rR,rY]]

    # rest of the layers
    parHv = [parH[1:], V-pV, R-pR, Y-pY]

    return pP_, parHv

# not updated:
def form_pP_(pP_, parHv, fd):  # fixed H nesting: aggH( subH( derH( parttv_ ))), pPs: >ave param clusters, nested
    '''
    p_sets with nesting depth, Hv is H, valt,rdnt,dect:
    aggHv: [aggH=subHv_, valt, rdnt, dect],
    subHv: [subH=derHv_, valt, rdnt, dect, 2],
    derHv: [derH=parttv_, valt, rdnt, dect, extt, 1]
    parttv: [[mtuple, dtuple],  valt, rdnt, dect, 0]
    '''

    # 1st layer initialization where pP_ is empty
    if not pP_:
        pP_, (parH, rV, rR, rY) = init_parHv(parHv[0], parHv[1], parHv[2], parHv[3], fd)
    else:
        parH, rV,rR,rY = parHv  # uncompressed H vals
        V,R,Y = 0,0,0  # compressed param sets:

    parH = copy(parH); part_ = []
    _play_ = pP_[-1]  # node_ + combined pars
    L = 1
    while len(parH) > L:  # get next player: len = sum(len lower lays): 1,1,2,4.: for subH | derH, not aggH?
        hL = 2 * L
        play_ = parH[L:hL]  # each player is [sub_pH, valt, rdnt, dect]
        # add conditionally compressed layers within layer_:
        pP_ += [form_pP_([], [play_,V,R,Y], fd)] if L > 2 else [play_]

        # compress current play_
        V,R,Y, rV, rR, rY = compress_play_(pP_, play_, part_, (V, R, Y ),(rV, rR, rY), fd)

        # compare compressed layer
        for _play in _play_:
            for play in play_:
                comp_pP(_play, play)
        L = hL

    if part_:
        pP_ += [[part_,V,R,Y]]; rV+=V; rR+=R; rY+=Y

    return [pP_,rV,rR,rY]  # root values


def compress_play_(pP_, play_, part_, rVals, Vals,  fd):

    V, R, Y = Vals
    rV, rR, rY = rVals

    for play in play_:  # 3-H unpack:
        if play[-1]:  # derH | subH
            if play[-1]>1:   # subH
                sspH,val,rdn,dec = play[0], play[1][fd], play[2][fd], play[3][fd]
                if val > ave:  # recursive eval,unpack
                    V+=val; R+=rdn; Y+=dec  # sum with sub-vals:
                    sub_pP_t = form_pP_([], [sspH,val,rdn,dec], fd)
                    part_ += [[sspH, sub_pP_t]]
                else:
                    if V:  # empty sub_pP_ terminates root pP
                        pP_ += [[part_,V,R,Y]]; rV+=V; rR+=R; rY+=Y  # root params
                        part_,V,R,Y = [],0,0,0  # pP params
                        # reset
            else:
                derH, val,rdn,dec,extt = play[0], play[1][fd], play[2][fd], play[3][fd], play[4]
                form_tuplet_pP_(extt, [pP_,rV,rR,rY], [part_,V,R,Y], v=0)
                sub_pP_t = form_pP_([], [derH,val,rdn,dec], fd)  # derH
                part_ += [[derH, sub_pP_t]]
        else:
            form_tuplet_pP_(play, [pP_,rV,rR,rY], [part_,V,R,Y], v=1)  # derLay

    return V, R, Y, rV, rR, rY


def comp_pP(_play, play):
    pass

def form_pP_recursive(parHv, fd):  # indefinite H nesting: (..HHH( HH( H( parttv_))).., init HH = [H] if len H > max

    parH, rV,rR,rY = parHv  # uncompressed summed G vals
    parP_ = []  # pPs: >ave param clusters, nested
    V,R,Y = 0,0,0
    parH = copy(parH); part_ = []; _player = parH[0]  # parP = [_player]  # node_ + combined pars
    L = 1
    while len(parH) > L:  # get next player of len = sum(len lower lays): 1,1,2,4., not for top H?
        hL = 2 * L
        play = parH[L:hL]  # [sparH, svalt, srdnt, sdect, depth]
        if play[-1]:       # recursive eval,unpack:
            subH,val,rdn,dec = play[0], play[1][fd], play[2][fd], play[3][fd]
            if val > ave:
                V+=val; R+=rdn; Y+=dec  # sum with sub-vals:
                sub_pP_t = form_pP_([subH,val,rdn,dec], fd)
                part_ += [[subH, sub_pP_t]]
            else:
                if V:  # empty sub_pP_ terminates root pP
                    parP_ += [[part_,V,R,Y]]; rV+=V; rR+=R; rY+=Y  # root params
                    part_ = [],V,R,Y = 0,0,0  # pP params
                    # reset parP
        else: form_tuplet_pP_(play, [parP_,rV,rR,rY], [part_,V,R,Y], v=1)  # derLay
        L = hL
    if part_:
        parP_ += [[part_,V,R,Y]]; rV+=V; rR+=R; rY+=Y
    return [parP_,rV,rR,rY]  # root values


def comp_parH(_lay,lay):  # nested comp between and then within same-fork dertuples?
    pass

def form_tuplet_pP_(ptuplet, part_P_v, part_v, v):  # ext or ptuple, params=vals

    part_P_,rVal,rRdn,rDec = part_P_v  # root params
    part_,Val,Rdn,Dec = part_v  # pP params
    if v: ptuplet, valt,rdnt,dect,_ = ptuplet

    valP_t = [[form_val_pP_(ptuple) for ptuple in ptuplet if sum(ptuple) > ave]]
    if valP_t:
        part_ += [valP_t]  # params=vals, no sum-> Val,Rdn,Max?
    else:
        if Val:  # empty valP_ terminates root pP
            part_P_ += [[part_, Val, Rdn, Dec]]
            part_P_v[1:] = rVal+Val, rRdn+Rdn, rDec+Dec  # root values
        part_v[:] = [],0,0,0  # reset

def form_val_pP_(ptuple):
    parP_ = []
    parP = [ptuple[0]] if ptuple[0] > ave else []  # init, need to use param type ave instead

    for par in ptuple[1:]:
        if par > ave: parP += [par]
        else:
            if parP: parP_ += [parP]  # terminate parP
            parP = []

    if parP: parP_ += [parP]  # terminate last parP
    return parP_  # may be empty

# draft
def sub_recursion(rroot, root, G_, lenH, fd, nrng=1):  # separate | interlaced fd recursion?

    # rng+|der+ over same-root nodes, forming multiple (Gm_,Gd_) GG_tH:
    GG_tH, out_tH = [], []
    val_t, rdn_t = [[],[]],[[],[]]

    while True:
        # not revised:
        Et = [[0,0],[0,0],[0,0]]  # grapht link_' eValt, eRdnt, eDect(currently not used)
        if fd:  # der+
            for link in root.link_:  # reform links
                if link.Vt[1] < G_aves[1] * link.Rt[1]: continue  # maybe weak after rdn incr?
                comp_G(link._G, link.G, link, Et, lenH)
        else:  # rng+
            for i, _G in enumerate(G_):  # form new link_ from original node_
                for G in G_[i + 1:]:
                    dy = _G.box.cy - G.box.cy; dx = _G.box.cx - G.box.cx
                    if np.hypot(dy, dx) < 2 * nrng:  # max distance between node centers, init=2
                        link = CderG(_G=_G, G=G)
                        comp_G(_G, G, link, Et, lenH)

        GG_t, valt, rdnt = form_graph_t(root, G_, Et, nrng)  # root_fd, eval sub+, feedback per graph
        if GG_t:
            GG_tH += [GG_t]
        else: break  # maybe empty
        for i in 0,1:
            val_t[i] += [valt[i]]; rdn_t[i] += [rdnt[i]+1]  # 1 is process redundancy to lower sub+
        if valt[fd] < G_aves[fd] * rdnt[fd]:
            break
        val_ = sorted(val_t[fd])  # max val in val_[0]?
        out_tH = []  # this will be reset per while loop, while GG_tH will accumulate more GG_t per while loop?
        for i, val in enumerate(val_):
            # rough pseudocode:
            if val > G_aves[fd] * (rdn_t[i][fd]+i):  # also remove init rdn?
                out_tH += [GG_tH[i][fd]]  # should be GG_tH to get each GG_t?
            else:
                break

    return out_tH



def node_connect(_G_):  # node connectivity = sum surround link vals, incr.mediated: Graph Convolution of Correlations
    '''
    Aggregate direct * indirect connectivity per node from indirect links via associated nodes, in multiple cycles.
    Each cycle adds contributions of previous cycles to linked-nodes connectivity, propagated through the network.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png
    '''
    while True:
        # eval accumulated G connectivity vals, indirectly extending their range
        G_ = []  # next connectivity expansion, more selective by DVt,Lent = [0,0],[0,0]?
        for G in _G_:
            uprimt = [[],[]]  # >ave updates of direct links
            for i in 0,1:
                val,rdn,dec = G.Vt[i],G.Rt[i],G.Dt[i]  # connect by last layer
                ave = G_aves[i]
                for link in G.Rim_tH[-1][i]:
                    lval,lrdn,ldec = link.Vt[i],link.Rt[i],link.Dt[i]
                    _G = link._G if link.G is G else link.G
                    _val,_rdn,_dec = _G.Vt[i],_G.Rt[i],_G.Dt[i]
                    # Vt.. for segment_node_:
                    V = ldec * (val+_val); dv = V-link.Vt[i]; link.Vt[i] = V
                    R = ldec * (rdn+_rdn); dr = R-link.Rt[i]; link.Rt[i] = R
                    D = ldec * (dec+_dec); dd = D-link.Dt[i]; link.Dt[i] = D
                    if dv > ave * dr:
                        G.Vt[i]+=V; G.Rt[i]+=R; G.Dt[i]+=D  # add link last layer vals
                        uprimt[i] += [link]  # dVt[i] += dv; L = len(uprimt[i]); Lent[i] += L for more selective eval
                    if V > ave * R:
                        G.evalt[i] += dv; G.erdnt[i] += dr; G.edect[i] += dd
            if any(uprimt):  # pruned for next loop
                G.Rim_tH[-1] = uprimt

        if G_: _G_ = G_  # exclude weakly incremented Gs from next connectivity expansion loop
        else:  break


def segment_node_(root, root_G_, fd, nrng):  # eval rim links with summed surround vals for density-based clustering

    # graph += [node] if >ave (surround connectivity * relative value of link to any internal node)
    igraph_ = []; ave = G_aves[fd]

    for G in root_G_:   # init per node, last-layer Vt,Vt,Dt:
        grapht = [[G],[], G.Vt,G.Rt,G.Dt, copy(G.rim_tH[-1][fd])]
        G.roott[fd] = grapht  # roott for feedback
        igraph_ += [grapht]
    _graph_ = igraph_
    while True:
        graph_ = []
        for grapht in _graph_:  # extend grapht Rim with +ve in-root links
            G_, Link_, Valt, Rdnt, Dect, Rim = grapht
            inVal, inRdn = 0,0  # new in-graph +ve
            new_Rim = []
            for link in Rim:  # unique links
                if link.G in G_:
                    G = link.G; _G = link._G
                else:
                    G = link._G; _G = link.G
                if _G in G_: continue
                # connect by rel match of nodes * match of node Vs: surround M|Ds,
                # V = how deeply inside the graph is G
                cval = link.Vt[fd] + get_match(G.Vt[fd],_G.Vt[fd])  # same coef for int and ext match?
                crdn = link.Rt[fd] + (G.Rt[fd] + _G.Rt[fd]) / 2
                if cval > ave * crdn:  # _G and its root are effectively connected
                    # merge _root:
                    _grapht = _G.roott[fd]
                    _G_,_Link_,_Valt,_Rdnt,_Dect,_Rim = _grapht
                    Link_[:] = list(set(Link_+_Link_)) + [link]
                    for g in _G_:
                        g.roott[fd] = grapht
                        if g not in G_: G_+=[g]
                    for i in 0,1:
                        Valt[i]+=_Valt[i]; Rdnt[i]+=_Rdnt[i]; Dect[i]+=_Dect[i]
                        inVal += _Valt[fd]; inRdn += _Rdnt[fd]
                    if _grapht in igraph_:
                        igraph_.remove(_grapht)
                    new_Rim += [link for link in _Rim if link not in new_Rim+Rim+Link_]
            # for next loop:
            if len(new_Rim) * inVal > ave * inRdn:
                graph_ += [[G_,Link_, Valt,Rdnt,Dect, new_Rim]]

        if graph_: _graph_ = graph_  # selected graph expansion
        else: break
    # -> Cgraphs if Val > ave * Rdn:
    return [sum2graph(root, graph, fd, nrng) for graph in igraph_ if graph[2][fd] > ave * (graph[3][fd])]



def sum2graph(root, grapht, fd, nrng):  # sum node and link params into graph, aggH in agg+ or player in sub+

    G_,Link_,Vt,Rt,Dt,_ = grapht  # last-layer vals only; depth 0:derLay, 1:derHv, 2:subHv

    graph = Cgraph(fd=fd, node_=[G_], L=len(G_),link_=Link_,Vt=Vt, Rt=Rt, Dt=Dt, rng=nrng)
    graph.roott[fd] = root
    for link in Link_:
        link.roott[fd]=graph
    eH, valt,rdnt,dect, evalt,erdnt,edect = [], [0,0],[0,0],[0,0], [0,0],[0,0],[0,0]  # grapht int = node int+ext
    A0, A1, S = 0,0,0
    for G in G_:
        for i, link in enumerate(G.rim_tH[-1][fd]):
            if i: sum_derHv(G.esubH[-1], link.subH[-1], base_rdn=link.Rt[fd])  # [derH, valt,rdnt,dect,extt,1]
            else: G.esubH += [deepcopy(link.subH[-1])]  # link.subH: cross-der+) same rng, G.esubH: cross-rng?
            for j in 0,1:
                G.evalt[j]+=link.Vt[j]; G.erdnt[j]+=link.Rt[j]; G.edect[j]+=link.Dt[j]
        graph.box += G.box
        graph.ptuple += G.ptuple
        sum_derH([graph.derH,[0,0],[1,1]], [G.derH,[0,0],[1,1]], base_rdn=1)
        sum_subHv([eH,evalt,erdnt,edect,2], [G.esubH,G.evalt,G.erdnt,G.edect,2], base_rdn=G.erdnt[fd])
        sum_aggHv(graph.aggH, G.aggH, base_rdn=1)
        A0 += G.A[0]; A1 += G.A[1]; S += G.S
        for j in 0,1:
            evalt[j] += G.evalt[j]; erdnt[j] += G.erdnt[j]; edect[j] += G.edect[j]
            valt[j] += G.valt[j]; rdnt[j] += G.rdnt[j]; dect[j] += G.dect[j]

    graph.aggH += [[eH,evalt,erdnt,edect,2]]  # new derLay
    for i in 0,1:
        graph.valt[i] = valt[i]+evalt[i]  # graph internals = G Internals + Externals
        graph.rdnt[i] = rdnt[i]+erdnt[i]
        graph.dect[i] = dect[i]+edect[i]
    graph.A = [A0,A1]; graph.S = S

    if fd:
        for link in graph.link_:  # assign alt graphs from d graph, after both linked m and d graphs are formed
            mgraph = link.roott[0]
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]
                        for i in 0,1:
                            G.avalt[i] += alt_G.valt[i]; G.ardnt[i] += alt_G.rdnt[i]; G.adect[i] += alt_G.dect[i]

    return graph


def form_graph_t(root, G_, Et, nrng):

    _G_ = [G for G in G_ if len(G.rim_tH)>len(root.rim_tH)]  # prune Gs unconnected in current layer

    node_connect(_G_)  # Graph Convolution of Correlations over init _G_
    GG_t, valt, rdnt = [[],[]], [0,0], [1,1]
    for fd in 0,1:
        if Et[0][fd] > ave * Et[1][fd]:  # eValt > ave * eRdnt, else no clustering, keep root.node_
            graph_ = segment_node_(root, _G_, fd, nrng)  # fd: node-mediated Correlation Clustering
            GG_t[fd] = graph_  # may be empty
    if any(GG_t): G_[:] = GG_t

    return GG_t, valt, rdnt
   

# probably just form_graph_t that returns GG_t, valt, rdnt, without agg+ call?
# not revised:
'''
def form_graph_t(root, G_, Et, nrng):

    _G_ = [G for G in G_ if len(G.rim_tH)>len(root.rim_tH)]  # prune Gs unconnected in current layer

    node_connect(_G_)  # Graph Convolution of Correlations over init _G_
    node_t = []
    for fd in 0,1:
        if Et[0][fd] > ave * Et[1][fd]:  # eValt > ave * eRdnt, else no clustering, keep root.node_
            graph_ = segment_node_(root, _G_, fd, nrng)  # fd: node-mediated Correlation Clustering
            if not graph_: continue
            for graph in graph_:  # eval sub+ per node
                if graph.Vt[fd] * (len(graph.node_[-1])-1)*root.rng > G_aves[fd] * graph.Rt[fd]:

                    # current depth sub+
                    agg_recursion(root, graph, graph.node_[-1], len(graph.aggH[-1][0]), fd, nrng+1*(1-fd))  # nrng+ if not fd

                    # higher layer's sub+
                    rroot = graph
                    rfd = rroot.fd
                    while isinstance(rroot.roott, list) and rroot.roott[rfd]:  # not blob
                        rroot = rroot.roott[rfd]
                        Val, Rdn = 0, 0
                        if isinstance(rroot.node_[-1][0], list):  # node_ is node_t
                            node_ = rroot.node_[-1][rfd]
                        else:
                            node_ = rroot.node_[-1]

                        for node in node_:  # sum vals and rdns from all higher nodes
                            Rdn += node.rdnt[rfd]
                            Val += node.valt[rfd]
                        # include rroot.Vt and Rt?
                        if Val * (len(rroot.node_[-1])-1)*rroot.rng > G_aves[fd] * Rdn:
                            # not sure about nrg here
                            agg_recursion(root, graph, rroot.node_[-1], len(rroot.aggH[-1][0]), rfd, nrng+1*(1-rfd))  # nrng+ if not fd

                else:
                    root.fback_t[root.fd] += [[graph.aggH, graph.valt, graph.rdnt, graph.dect]]
                    feedback(root,root.fd)  # update root.root..
            node_t += [graph_]  # may be empty
        else: node_t += []
    if any(node_t): G_[:] = node_t
'''

def feedback(root, fd):  # called from form_graph_, append new der layers to root

    AggH, ValHt, RdnHt, DecHt = deepcopy(root.fback_t[fd].pop(0))  # init with 1st tuple
    while root.fback_t[fd]:
        aggH, valHt, rdnHt, decHt = root.fback_t[fd].pop(0)
        sum_aggH(AggH, aggH, base_rdn=0)
    sum_aggH(root.aggH,AggH, base_rdn=0)

    if isinstance(root, Cgraph) and root.root:  # root is not CEdge, which has no roots
        rroot = root.root
        fd = root.fd  # current node_ fd
        fback_ = rroot.fback_t[fd]
        fback_ += [[AggH, ValHt, RdnHt, DecHt]]
        if fback_ and (len(fback_) == len(rroot.node_t)):  # flat, all rroot nodes terminated and fed back
            # getting cyclic rroot here not sure why it can happen, need to check further
            feedback(rroot, fd)  # sum2graph adds aggH per rng, feedback adds deeper sub+ layers


# more selective: only for parallel clustering?
def select_init_(Gt_, fd):  # local max selection for sparse graph init, if positive link

    init_, non_max_ = [],[]  # pick max in direct links, no recursively mediated links max: discontinuous?

    for node, val in Gt_:
        if node in non_max_: continue  # can't init graph
        if val<=0:  # no +ve links
            if sum(node.val_Ht[fd]) > ave * sum(node.rdn_Ht[fd]):
                init_+= [[node, 0]]  # single-node proto-graph
            continue
        fmax = 1
        for link in node.link_:
            _node = link.G if link._G is node else link._G
            if val > Gt_[_node.it[fd]][1]:
                non_max_ += [_node]  # skip as next node
            else:
                fmax = 0; break  # break is not necessary?
        if fmax:
            init_ += [[node,val]]
    return init_