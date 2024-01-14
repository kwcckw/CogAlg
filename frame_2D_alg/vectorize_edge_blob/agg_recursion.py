import numpy as np
from copy import deepcopy, copy
from itertools import combinations, zip_longest
from .classes import Cgraph, CderG, Cmd
from .filters import aves, ave_mL, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_derH, sum_derH, comp_ptuple, sum_dertuple, comp_dtuple, get_match

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
Clustering criterion is G M|D, summed across >ave vars if selective comp (<ave vars are not compared, so they don't add costs).
Fork selection should be per var or co-derived der layer or agg level. 
There are concepts that include same matching vars: size, density, color, stability, etc, but in different combinations.
Weak value vars are combined into higher var, so derivation fork can be selected on different levels of param composition.

Clustering by variance: lend|borrow, contribution or counteraction to similarity | stability, such as metabolism? 
-
G is graph:
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

def vectorize_root(blob, verbose):  # vectorization in 3 composition levels of xcomp, cluster:

    edge, adj_Pt_ = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering

    comp_P_(edge, adj_Pt_)  # vertical, lateral-overlap P cross-comp -> PP clustering

    for fd, node_ in enumerate(edge.node_):  # always node_t
        if edge.valt[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.rdnt[fd]:

            for PP in node_: PP.roott = [None, None]
            # discontinuous PP cross-comp, cluster -> G_t
            agg_recursion(None, edge, node_, nrng=1)  # agg+, no Cgraph nodes

    return edge


def agg_recursion(rroot, root, node_, nrng=1, lenH=None, lenHH=None):  # lenH = len(root.aggH[-1][0]), lenHH: same in agg_compress

    Et = [[0,0],[0,0],[0,0]]  # grapht link_' eValt, eRdnt, eDect(currently not used)
    fd = not nrng  # compositional agg|sub recursion in root graph:

    if fd:  # der+
        for link in root.link_:  # reform links
            if link.Vt[1] > G_aves[1]*link.Rt[1]:  # maybe weak after rdn incr?
                comp_G(link, Et, lenH, lenHH)
    else:   # rng+
        for _G, G in combinations(node_, r=2):  # form new link_ from original node_
            dy = _G.box.cy - G.box.cy; dx = _G.box.cx - G.box.cx
            if np.hypot(dy,dx) < 2 * nrng:  # max distance between node centers, init=2
                link = CderG(_G=_G, G=G)
                comp_G(link, Et, lenH, lenHH)

    form_graph_t(root, node_, Et, nrng)  # root_fd, eval sub+, feedback per graph
    if isinstance(node_[0],list):  # node_t was formed above

        for i, G_ in enumerate(node_):
            if root.valt[i] * (len(G_)-1)*root.rng > G_aves[i] * root.rdnt[i]:
                # agg+/ node_( sub)agg+/ node, vs sub+ only in comp_slice
                agg_recursion(rroot, root, G_, nrng=1)  # der+ if fd, else rng+ =2
                if rroot:
                    rroot.fback_t[i] += [[root.aggH,root.valt,root.rdnt,root.dect]]
                    feedback(rroot,i)  # update root.root..


def form_graph_t(root, G_, Et, nrng, lenH=None, lenHH=None):  # form Gm_,Gd_ from same-root nodes

    # select Gs connected in current layer:
    lenH_root = 0 if root.rim_t[1] == 0 else len(root.rim_t[0])  # if rim_t[1] (depth) is 0 , the lenH of root should stay 0?
    _G_ = [G for G in G_ if G.rim_t[1] > 0 and len(G.rim_t[0])> lenH_root]  # skip Gs with rim_t's depth of 0 too?? 


    node_connect(_G_, lenHH!=None)  # Graph Convolution of Correlations over init _G_
    node_t = []
    for fd in 0,1:
        if Et[0][fd] > ave * Et[1][fd]:  # eValt > ave * eRdnt: cluster
            graph_ = segment_node_(root, _G_, fd, nrng, lenH, lenHH)  # fd: node-mediated Correlation Clustering
            for graph in graph_:
                # eval sub+ per node
                if graph.Vt[fd] * (len(graph.node_)-1)*root.rng > G_aves[fd] * graph.Rt[fd]:
                    node_ = graph.node_  # flat in sub+
                    if lenH: lenH = len(node_[0].esubH[-lenH:])  # in agg_compress
                    else:    lenH = len(graph.aggH[-1][0])  # in agg_recursion
                    nrng = 0 if fd else nrng+1
                    # section below pending update, if frd, it should call agg_compress instead
                    # agg_recursion(root, graph, node_, nrng, lenH, lenHH)
                else:
                    root.fback_t[root.fd] += [[graph.aggH, graph.valt, graph.rdnt, graph.dect]]
                    # feedback pending update
                    # feedback(root,root.fd)  # update root.root..
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]
    if any(node_t):
        G_[:] = node_t  # else keep root.node_


def node_connect(_G_, frd):  # node connectivity = sum surround link vals, incr.mediated: Graph Convolution of Correlations
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
                for link in unpack_rim(G.rim_t, i):
                    # >ave derG in fd rim
                    lval,lrdn,ldec = link.Vt[i],link.Rt[i],link.Dt[i]
                    _G = link._G if link.G is G else link.G
                    _val,_rdn,_dec = _G.Vt[i],_G.Rt[i],_G.Dt[i]
                    # Vt.. for segment_node_:
                    V = ldec * (val+_val); dv = V-link.Vt[i]; link.Vt[i] = V
                    R = ldec * (rdn+_rdn); dr = R-link.Rt[i]; link.Rt[i] = R
                    D = ldec * (dec+_dec); dd = D-link.Dt[i]; link.Dt[i] = D
                    if dv > ave * dr:
                        G.Vt[i]+=V; G.Rt[i]+=R; G.Dt[i]+=D  # add link last layer vals
                        uprimt[i] += [link]
                        # more selective eval: dVt[i] += dv; L = len(uprimt[i]); Lent[i] += L
                    if V > ave * R:
                        G.evalt[i] += dv; G.erdnt[i] += dr; G.edect[i] += dd
            if any(uprimt):  # pruned for next loop
                if frd:
                    # revise depending on lenHH: different nesting?
                    for i in 0,1: 
                        rim = unpack_rim(G.rim_t, i)
                        rim[:] = uprimt[i]
                    else: G.rim_t[:] = uprimt
                G_ += [G]

        if G_: _G_ = G_  # exclude weakly incremented Gs from next connectivity expansion loop
        else:  break

def unpack_rim(rim_t, fd):  # if init rim_t is [[[],[]],0]

    if rim_t[1] == 2:  # depth=2 is rim_tH in agg++(agg_cpr)
        rim_t = rim_t[0][-1]  # last rim_t
    if rim_t[1] == 1:
        if isinstance(rim_t[0][-1],list):  # rim_t[0] is [rim_t2, rim_t2, ...], so we still need [-1] to select the last rim_t
              rim = rim_t[0][-1][fd][-1]  # rim_t in agg++
        else: rim = rim_t[0][-1][fd]  # rimtH in agg+
    else:
        rim = rim_t[fd]  # base rimt

    return rim


def segment_node_(root, root_G_, fd, nrng, lenH=None, lenHH=None):  # eval rim links with summed surround vals for density-based clustering

    # graph += [node] if >ave (surround connectivity * relative value of link to any internal node)
    igraph_ = []; ave = G_aves[fd]

    for G in root_G_:   # init per node,  last-layer Vt,Vt,Dt:
        grapht = [[G],[], G.Vt,G.Rt,G.Dt, copy(unpack_rim(G.rim_t, fd))]  # init link_ with rim
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
    return [sum2graph(root, graph, fd, nrng, lenH, lenHH) for graph in igraph_ if graph[2][fd] > ave * graph[3][fd]]


def sum2graph(root, grapht, fd, nrng, lenH=None, lenHH=None):  # sum node and link params into graph, aggH in agg+ or player in sub+

    G_,Link_,Vt,Rt,Dt,_ = grapht  # last-layer vals only; depth 0:derLay, 1:derHv, 2:subHv

    graph = Cgraph(fd=fd, node_=G_, L=len(G_),link_=Link_,Vt=Vt, Rt=Rt, Dt=Dt, rng=nrng)
    graph.roott[fd] = root
    for link in Link_:
        link.roott[fd]=graph
    eH, valt,rdnt,dect, evalt,erdnt,edect = [], [0,0],[0,0],[0,0], [0,0],[0,0],[0,0]  # grapht int = node int+ext
    A0, A1, S = 0,0,0
    for G in G_:
        unpack_subLay(G, fd, lenH, lenHH, i=0)
        ''' use unfinished unpack_derLay
        if frd:  # from agg_cpr, call unpack_derLay recursively per lenH if lenHH, to sum all links of rim_
        old:
            rdHt = [[[[]],[[]]], 1]  # G.esubH[-1]  (init with 1st empty derHv)
            G.esubH += [rdHt]
        else:
            derH = [[],1]  # G.esubH
            for link in rim:  # sum last rd+ esubH layer
                if len(link.subH) > len(G.esubH):  # link was appended in sub+
                    sum_subHv(derH, link.subH[-1], base_rdn=link.Rt[fd])  # [derH,valt,rdnt,dect,extt,1] per sub+
                    for j in 0,1:
                        G.evalt[j]+=link.Vt[j]; G.erdnt[j]+=link.Rt[j]; G.edect[j]+=link.Dt[j]
            G.esubH += [derH]  # link.subH: cross-der+) same rng, G.esubH: cross-rng?
        '''
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
    
    # graph internals = G Internals + Externals
    graph.valt = Cmd(*valt) + evalt  
    graph.rdnt = Cmd(*rdnt) + erdnt
    graph.dect = Cmd(*dect) + edect

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


# draft, use in sum2grapht:
def unpack_subLay(G, fd, lenH, lenHH, i):  # i to recursively call deeper layers of rdH if lenHH, to sum all links of rim_

    for link in unpack_rim(G.rim_t, fd):  # sum last rd+ esubH layer
        if lenHH != None:  # agg_cpr's lenHH is 0             
            # link.subH is [ rdHt1, rdHt2, ...] 
            # each rdnHt is [rdt, depth]
            # each rdt is [msubH_, dsubH_]
            # [-1] select last rdHt, [0] to skip depth, [fd] to select fork, [-1] to select last subH
            subH = link.subH[-1][0][fd][-1]  # from sub+'agg++ (this should be select from i too？)
        else:     subH = link.subH[0]  # from agg+ or no-sub+'agg++

        if len(subH) > (lenH or 0):  # was appended in last xcomp, deeper nesting in agg++ link.subH
            sum_derHv(G.esubH[-1], subH[i], base_rdn=link.Rt[fd])
            G.evalt[fd] += link.Vt[fd]; G.erdnt[fd] += link.Rt[fd]; G.edect[fd] += link.Dt[fd]

        if lenHH>0:
            unpack_subLay(G, fd, lenH, lenHH, i+1)

def comp_G(link, Et, lenH=None, lenHH=None, fdcpr=0):  # lenH in sub+|rd+, lenHH in agg_compress sub+ only

    _G, G = link._G, link.G
    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0, 1,1, 0,0

    # keep separate P ptuple and PP derH, empty derH in single-P G, + empty aggH in single-PP G:
    # / P:
    mtuple, dtuple, Mtuple, Dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1, fagg=1)
    mval, dval = sum(mtuple), sum(abs(d) for d in dtuple)  # mval is signed, m=-min in comp x sign
    mrdn = dval>mval; drdn = dval<=mval
    dect = [0,0]
    for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
        for i, (par, max, ave) in enumerate(zip(ptuple, Ptuple, aves)):
            # compute link decay coef: par/ max(self/same)
            if fd: dect[fd] += par/max if max else 1
            else:  dect[fd] += (par+ave)/ max if max else 1
    derLay0 = [[mtuple,dtuple],[mval,dval],[mrdn,drdn],[dect[0]/6,dect[1]/6], 0]  # ave of 6 params
    Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn; Mdec+=dect[0]/6; Ddec+=dect[1]/6

    # / PP:
    _derH,derH = _G.derH,G.derH
    if _derH and derH:  # empty in single-node Gs
        dderH = []
        for _lay, lay in zip(_derH,derH):
            mtuple,dtuple, Mtuple,Dtuple = comp_dtuple(_lay[1], lay[1], rn=1, fagg=1)
            mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
            mrdn = dval > mval; drdn = dval < mval
            mdec, ddec = 0, 0
            for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
                for (par, max, ave) in zip(ptuple, Ptuple, aves):  # different ave for comp_dtuple
                    if fd: ddec += par/max if max else 1
                    else:  mdec += (par+ave)/(max) if max else 1
            mdec /= 6; ddec /= 6
            Mval+=dval; Dval+=mval; Mdec=(Mdec+mdec)/2; Ddec=(Ddec+ddec)/2
            dderH += [[[mtuple,dtuple], [mval,dval],[mrdn,drdn],[mdec,ddec], 0]]
    else:
        dderH = []
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec])
    derHv = [[derLay0]+dderH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec], der_ext, 1]  # appendleft derLay0 from comp_ptuple
    SubH = [derHv]  # init layers of SubH, higher layers added by comp_aggH:

    # / G:
    if link.subH:  # old link: not empty aggH
        subH, valt,rdnt,dect = comp_aggHv(_G.aggH, G.aggH, rn=1)
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
        Mdec = (Mdec+dect[0])/2; Ddec = (Ddec+dect[1])/2
        if lenHH == None:
            link.subH = SubH+subH  # concat higher derHvs
        else:  # tentative: subH[rdHt][fd] + subH:
            link.subH[-1][0][fdcpr] += [SubH + subH]  # call from rd+: rim_ and corresponding ssubH are fd-specific
    else:  # new link
        if lenHH == None:
            link.subH = SubH
        else:
            # rdt = [[[msubH1 msubH2,..],[dsubH1, dsubH2,...]], 1]
            # rdHt = [[[[msubH1 msubH2,..],[dsubH1, dsubH2,...]], 1]]  (added bracket)
            link.subH =  [[[[[]],[SubH]], 1]] if fdcpr else [[[[SubH],[[]]], 1]]



    link.Vt,link.Rt,link.Dt = Valt,Rdnt,Dect = [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]  # reset per comp_G

    for fd, (Val,Rdn,Dec) in enumerate(zip(Valt,Rdnt,Dect)):
        if Val > G_aves[fd] * Rdn:
            # eval fork grapht in form_graph_t:
            Et[0][fd] += Val; Et[1][fd] += Rdn; Et[2][fd] += Dec
            append_rim(link, lenH, lenHH, Val,Rdn,Dec, fd)

# draft
def append_rim(link, lenH, lenHH, Val,Rdn,Dec, fd):  # fmin: call from base agg+

    for G in link._G, link.G:
        rim_t = G.rim_t
        # rim_t nesting: [[],0] | [rimt|rim_t, 1] | [rim_tH, 2] | [rim_tHH, 3]
        root_depth = (lenH != None) + (lenHH != (None or 0)) + 1  # lenHH is None in base agg+, maybe 0 in agg_cpr

        if lenHH == None:  # base agg+
            if rim_t[1] == root_depth:  # root_depth and rim_t not incremented yet
                # add link layer:
                G.rim_t[1] = 2  # max depth in base agg+
                if fd:
                    # we need extra bracket here on top of [],[link] t pack each layer rimt
                    G.rim_t[0] = [[[],[link]]]; G.Vt=[0,Val]; G.Rt=[0,Rdn]; G.Dt=[0,Dec]
                else:
                    G.rim_t[0] = [[[link],[]]]; G.Vt=[Val,0]; G.Rt=[Rdn,0]; G.Dt=[Dec,0]
            else:
                # append last link layer:
                rim_t[0][-1][fd] += [link];  G.Vt[fd] += Val; G.Rt[fd] += Rdn; G.Dt[fd] += Dec
        else:
            # agg_compress rd+
            depth = G.rim_t[1]; ddepth = root_depth - depth
            if depth == 0:
                # add rim_t layer, extra bracket vs. rimts in base agg+:
                if fd:
                    rim_t = [[[]],[[link]]]; G.Vt=[0,Val]; G.Rt=[0,Rdn]; G.Dt=[0,Dec]
                else:
                    rim_t = [[[link]],[[]]]; G.Vt=[Val,0]; G.Rt=[Rdn,0]; G.Dt=[Dec,0]
                # nest rim_t to ddepth:
                for d in range(ddepth): rim_t = [[rim_t], d+1]  # +1: loop starts at 0
                G.rim_t = rim_t
            else:
                if depth == 1:
                    if len(G.rim_t[0])==(lenH or 0): G.rim_t[0][-1][fd] += [[link]]  # append rim_tH[fd] with rim_
                    else:                            G.rim_t[0][-1][fd][-1] += [link] # append last rim
                else:  # max ddepth == 2
                    # this looks wrong:
                    if len(G.rim_t[0])==(lenH or 0): G.rim_t[0][-1][0][-1][fd] += [[link]]
                    else:                            G.rim_t[0][-1][0][-1][fd][-1] += [link]
                G.Vt[fd] += Val; G.Rt[fd] += Rdn; G.Dt[fd] += Dec


def comp_aggHv(_aggH, aggH, rn):  # no separate ext

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    SubH = []

    for _lev, lev in zip(_aggH, aggH):  # compare common subHs, if lower-der match?
        if _lev and lev:
            dsubH, valt,rdnt,dect = comp_subHv(_lev[0],lev[0], rn)
            SubH += dsubH  # concat
            Mdec += dect[0]; Ddec += dect[1]
            mval,dval = valt; Mval += mval; Dval += dval
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + mval <= dval
    if SubH:
        S = min(len(_aggH),len(aggH)); Mdec/= S; Ddec /= S  # normalize

    return SubH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]

# pending update on subHv_t
def comp_subHv(_subH, subH, rn):

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    dsubH =[]

    for _lay, lay in zip(_subH, subH):  # compare common lower layer|sublayer derHs, if prior match?

        dderH, valt,rdnt,dect = comp_derHv(_lay[0],lay[0], rn)  # derHv: [derH, valt, rdnt, dect, extt, 1]:
        dextt = [comp_ext(_ext,ext,[Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]) for _ext,ext in zip(_lay[-2],lay[-2])]

        dsubH += [[dderH, valt,rdnt,dect,dextt, 1]]  # flat
        Mdec += dect[0]; Ddec += dect[1]
        mval,dval = valt; Mval += mval; Dval += dval
        Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval
    if dsubH:
        S = min(len(_subH),len(subH)); Mdec/= S; Ddec /= S  # normalize

    return dsubH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]  # new layer,= 1/2 combined derH


def comp_derHv(_derH, derH, rn):  # derH is a list of der layers or sub-layers, each is ptuple_tv

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    dderH =[]

    for _lay, lay in zip(_derH,derH):
        # comp dtuples, eval mtuples:
        mtuple,dtuple, Mtuple,Dtuple = comp_dtuple(_lay[0][1], lay[0][1], rn, fagg=1)
        mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
        mrdn = dval > mval; drdn = dval < mval
        dect = [0,0]
        for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
            for (par, max, ave) in zip(ptuple, Ptuple, aves):  # different ave for comp_dtuple
                if fd: dect[fd] += par/max if max else 1
                else:  dect[fd] += (par+ave)/(max) if max else 1
        dect[0]/=6; dect[1]/=6
        dderH += [[[mtuple,dtuple], [mval,dval],[mrdn,drdn],dect, 0]]
        Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn
        Mdec+=dect[0]; Ddec+=dect[1]

    if dderH:
        S = min(len(_derH),len(derH)); Mdec /= S; Ddec /= S  # normalize

    return dderH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]  # new derLayer,= 1/2 combined derH


def sum_aggHv(AggH, aggH, base_rdn):

    if aggH:
        if AggH:
            for Layer, layer in zip_longest(AggH,aggH, fillvalue=[]):
                sum_subHv(Layer, layer, base_rdn)
        else:
            AggH[:] = deepcopy(aggH)

# pending update on subHv_t
def sum_subHv(T, t, base_rdn, fneg=0):

    if t:
        if T:
            SubH,Valt,Rdnt,Dect,_ = T; subH,valt,rdnt,dect,_ = t
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i]+dect[i])/2
            if SubH:
                for Layer, layer in zip_longest(SubH,subH, fillvalue=[]):
                    sum_derHv(Layer, layer, base_rdn, fneg)  # _lay[0][0] is mL
            else:
                SubH[:] = deepcopy(subH)
        else:
            T[:] = deepcopy(t)


def sum_derHv(T,t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    if t:
        if T:
            DerH, Valt, Rdnt, Dect, Extt_,_ = T
            derH, valt, rdnt, dect, extt_,_ = t
            for Extt, extt in zip(Extt_,extt_):
                sum_ext(Extt, extt)
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i] + dect[i])/2
            DerH[:] = [
                [[sum_dertuple(Dertuple,dertuple, fneg*i) for i,(Dertuple,dertuple) in enumerate(zip(Tuplet,tuplet))],
                  [V+v for V,v in zip(Valt,valt)], [R+r+base_rdn for R,r in zip(Rdnt,rdnt)], [(D+d)/2 for D,d in zip(Dect,dect)], 0
                ]
                for [Tuplet,Valt,Rdnt,Dect,_], [tuplet,valt,rdnt,dect,_]  # ptuple_tv
                in zip_longest(DerH, derH, fillvalue=[([0,0,0,0,0,0],[0,0,0,0,0,0]), (0,0),(0,0),(0,0),0])
            ]

        else:
            T[:] = deepcopy(t)

def sum_ext(Extt, extt):

    if isinstance(Extt[0], list):
        for Ext,ext in zip(Extt,extt):  # ext: m,d tuple
            for i,(Par,par) in enumerate(zip(Ext,ext)):
                Ext[i] = Par+par
    else:  # single ext
        for i in range(3): Extt[i]+=extt[i]  # sum L,S,A


def comp_ext(_ext, ext, Valt, Rdnt, Dect):  # comp ds:

    (_L,_S,_A),(L,S,A) = _ext,ext
    dL = _L-L

    if isinstance(A,list):
        if any(A) and any(_A):
            mA,dA = comp_angle(_A,A); adA=dA
        else:
            mA,dA,adA = 0,0,0
        max_mA = max_dA = .5  # = ave_dangle
        dS = _S/_L - S/L  # S is summed over L, dS is not summed over dL
    else:
        mA = get_match(_A,A)- ave_dangle; dA = _A-A; adA = abs(dA); _aA=abs(_A); aA=abs(A)
        max_dA = _aA + aA; max_mA = max(_aA, aA)
        dS = _S - S
    mL = get_match(_L,L) - ave_mL
    mS = get_match(_S,S) - ave_mL

    m = mL+mS+mA; d = abs(dL)+ abs(dS)+ adA
    Valt[0] += m; Valt[1] += d
    Rdnt[0] += d>m; Rdnt[1] += d<=m
    _aL = abs(_L); aL = abs(L)
    _aS = abs(_S); aS = abs(S)

    # ave dec = ave (ave dec, ave L,S,A dec):
    Dect[0] = ((mL / max(aL,_aL) if aL or _aL else 1 +
                mS / max(aS,_aS) if aS or _aS else 1 +
                mA / max_mA if max_mA else 1) /3
                + Dect[0]) / 2
    Dect[1] = ((dL / (_aL+aL) if aL+_aL else 1 +
                dS / (_aS+aS) if aS+_aS else 1 +
                dA / max_dA if max_mA else 1) /3
                + Dect[1]) / 2

    return [[mL,mS,mA], [dL,dS,dA]]


def feedback(root, fd):  # called from form_graph_, append new der layers to root

    AggH, Valt, Rdnt, Dect = deepcopy(root.fback_t[fd].pop(0))
    # init with 1st tuple
    while root.fback_t[fd]:
        aggH, valt, rdnt, dect = root.fback_t[fd].pop(0)
        sum_aggHv(AggH, aggH, base_rdn=0)
        for i in 0,1:
            Valt[i] += valt[i]; Rdnt[i] += rdnt[i]; Dect[i] += dect[i]
            #-> root.fback_t
    if Valt[fd] > G_aves[fd] * Rdnt[fd]:  # or compress each level?
        root.aggH += AggH  # higher levels are not affected
        for j in 0,1:  # sum both in same root fork
            root.valt[fd] += Valt[j]; root.rdnt[fd] += Rdnt[j]; root.dect[fd] += Dect[j]

    if isinstance(root.roott, list):  # not Edge
        rroot = root.roott[fd]
        if rroot:
            fback_ = rroot.fback_t[fd]  # always node_t for feedback
            if fback_ and len(fback_) == len(rroot.node_[fd]):  # all nodes sub+ terminated
                feedback(rroot, fd)  # sum2graph adds higher aggH, feedback adds deeper aggH layers