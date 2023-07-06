import numpy as np
from copy import deepcopy, copy
from .classes import Cgraph
from .filters import aves, ave, ave_nsub, ave_sub, ave_agg, G_aves, med_decay, ave_distance, ave_Gm, ave_Gd
from .comp_slice import comp_angle, comp_aangle, comp_derH, sum_derH
from .sub_recursion import feedback  # temporary

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

# not fully updated
def agg_recursion(root):  # compositional recursion in root.PP_

    for i in 0,1: root.rdnt[0][i] += 1  # estimate, no node.rdnt[fd] += 1?

    comp_G_(root.node_, pri_G_=None, f1Q=1, fsub=0)  # cross-comp all Gs within rng
    graph_t = form_graph_(root)  # clustering via link_t
    # sub+:
    for fd, graph_ in enumerate(graph_t):
        if root.valt[0][fd] > ave_sub * root.rdnt[0][fd] and graph_:  # fixed costs and non empty graph_, same per fork
            sub_recursion_eval(root, graph_)
    # agg+:
    for fd, graph_ in enumerate(graph_t):
        if root.valt[0][fd] > G_aves[fd] * ave_agg * root.rdnt[0][fd] and len(graph_) > ave_nsub:
            agg_recursion(root)  # replace root.node_ with new graphs
        elif root.root_:  # if deeper agg+
            feedback(root, fd)  # update root.root..H, breadth-first

# draft:
def comp_G_(G_, pri_G_=None, f1Q=1, fd=0, fsub=0):  # cross-comp Graphs if f1Q, else comp G_s in comp_node_

    if not f1Q: dpars_=[]  # this was for nested node, we need single node with link-specific partial-parT access now

    for i, _iG in enumerate(G_ if f1Q else pri_G_):  # G_ is node_ of root graph, initially converted PPs
        # follow links in der+, loop all Gs (or link_?) in rng+:
        for iG in _iG.link_t[1] if fd \
            else G_[i+1:] if f1Q else G_:  # compare each G to other Gs in rng+, bilateral link assign, val accum:
            if not fd:   # not fd if f1Q?
                if iG in [node for link in _iG.link_ for node in link.node_]:  # the pair compared in prior rng+
                    continue
            dy = _iG.box[0]-iG.box[0]; dx = _iG.box[1]-iG.box[1]  # between center x0,y0
            distance = np.hypot(dy,dx) # Euclidean distance between centers, sum in sparsity, proximity = ave-distance
            if distance < ave_distance * ((np.sum(_iG.valt[1]) + np.sum(iG.valt[1])) / (2*sum(G_aves))):
                # same for cis and alt Gs:
                for _G, G in ((_iG, iG), (_iG.alt_Graph, iG.alt_Graph)):
                    if not _G or not G:  # or G.val
                        continue
                    # not sure here, should comparing internal derT[0] or external derT[1]?
                    # but for converted graph, their external derT is always empty
                    derT, valT, rdnT = [], [], []
                    for i in 0,1:
                        derH,valt,rdnt = comp_derH(_G.derT[i], G.derT[i], rn=1)  # comp layers while lower match?
                        derT += [derH]; valT+=[valt]; rdnT+=[rdnt] 
                    derG = Cgraph(node_=[_G,G], derT=derT,valt=valT,rdnt=rdnT, S=distance, A=[dy,dx], box=[])  # box is redundant to G
                    # add links:
                    _G.link_ += [derG]; G.link_ += [derG]  # no didx, no ext_valt accum?
                    # use both internal and external val? Or just external val?
                    if (valT[0][0] + valT[1][0]) > ave_Gm:
                        _G.link_t[0] += [derG]; G.link_t[0] += [derG]  # bi-directional
                    if (valT[0][1] + valT[1][1]) > ave_Gd:
                        _G.link_t[1] += [derG]; G.link_t[1] += [derG]

                    if not f1Q: dpars_ += [[derT,valT,rdnT]]  # comp G_s? not sure
                # implicit cis, alt pair nesting in mderH, dderH
    if not f1Q:
        return dpars_  # else no return, packed in links
    '''
    comp alts,val,rdn? cluster per var set if recurring across root: type eval if root M|D?
    '''

def form_graph_(root): # form derH in agg+ or sub-pplayer in sub+, G is node in GG graph

    G_ = root.node_
    mnode_, dnode_ = [],[]  # Gs with >0 +ve fork links:

    for G in G_:
        if G.link_t[0]: mnode_ += [G]  # all nodes with +ve links, not clustered in graphs yet
        if G.link_t[1]: dnode_ += [G]
    graph_t = []
    for fd, node_ in enumerate([mnode_, dnode_]):
        graph_ = []  # init graphs by link val:
        while node_:  # all Gs not removed in add_node_layer
            G = node_.pop(); gnode_ = [G]
            val = init_graph(gnode_, node_, G, fd, val=0)  # recursive depth-first gnode_ += [_G]
            graph_ += [[gnode_,val]]
        # prune graphs by node val:
        regraph_ = graph_reval_(graph_, [G_aves[fd] for graph in graph_], fd)  # init reval_ to start
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
        graph_t += [graph_]

    # add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
    return graph_t


def init_graph(gnode_, G_, G, fd, val):  # recursive depth-first gnode_+=[_G]

    for link in G.link_:
        # all positive links init graph, eval node.link_ in prune_node_layer
        _G = link.node_[1] if link.node_[0] is G else link.node_[0]
        if _G in G_:  # _G is not removed in prior loop
            gnode_ += [_G]
            G_.remove(_G)
            val += _G.valt[0][fd]
            val += init_graph(gnode_, G_, _G, fd, val)
    return val


def graph_reval_(graph_, reval_, fd):  # recursive eval nodes for regraph, after pruning weakly connected nodes

    regraph_, rreval_ = [],[]
    aveG = G_aves[fd]

    while graph_:
        graph,val = graph_.pop()
        reval = reval_.pop()  # each link *= other_G.aggH.valt

        if val > aveG:  # else graph is not re-inserted
            if reval < aveG:  # same graph, skip re-evaluation:
                regraph_+=[[graph,val]]; rreval_+=[0]
            else:
                regraph, reval = graph_reval([graph,val], fd)  # recursive depth-first node and link revaluation
                if regraph[1] > aveG:  # Val packed in regraph now
                    regraph_ += [regraph]; rreval_+=[reval]
    if rreval_:
        if max([reval for reval in rreval_]) > aveG:
            regraph_ = graph_reval_(regraph_, rreval_, fd)  # graph reval while min val reduction

    return regraph_

def graph_reval(graph, fd):  # exclusive graph segmentation by reval,prune nodes and links

    aveG = G_aves[fd]
    reval = 0  # reval proto-graph nodes by all positive in-graph links:

    for node in graph[0]:
        lval = 0  # link value
        _lval = node.valt[0][fd]  # = sum([link.valt[fd] for link in node.link_t[fd]])?
        for derG in node.link_t[fd]:
            val = derG.valt[0][fd]  # of top derH only
            _node = derG.node_[1] if derG.node_[0] is node else derG.node_[0]
            lval += val + (_node.valt[0][fd]-val) * med_decay
        reval += _lval - lval  # _node.link_t val was updated in previous round
    rreval = 0
    if reval > aveG:
        # prune:
        regraph, Val = [], 0  # reformed proto-graph
        # load Val from graph [1] instead?
        for node in graph[0]:
            val = node.valt[0][fd]
            if val < G_aves[fd] and node in graph:  # prune revalued node and its links
                for derG in node.link_t[fd]:
                    _node = derG.node_[1] if derG.node_[0] is node else derG.node_[0]
                    _link_ = _node.link_t[fd]
                    if derG in _link_: _link_.remove(derG)
                    rreval += derG.valt[0][fd] + (_node.valt[0][fd]-derG.valt[0][fd]) * med_decay  # else same as rreval += link_.val
            else:
                link_ = node.link_t[fd]  # prune node links only:
                remove_link_ = []
                for derG in link_:
                    _node = derG.node_[1] if derG.node_[0] is node else derG.node_[0]  # add med_link_ val to link val:
                    lval = derG.valt[0][fd] + (_node.valt[0][fd]-derG.valt[0][fd]) * med_decay
                    if lval < aveG:  # prune link, else no change
                        remove_link_ += [derG]
                        rreval += lval
                while remove_link_:
                    link_.remove(remove_link_.pop(0))
                Val += val
                regraph += [node]
        # recursion:
        if rreval > aveG and Val > aveG:
            regraph, reval = graph_reval([regraph,Val], fd)  # not sure about Val
            rreval += reval

    else: regraph, Val = graph

    return [regraph,Val], rreval
'''
In rng+, graph may be extended with out-linked nodes, merged with their graphs and re-segmented to fit fixed processors?
Clusters of different forks / param sets may overlap, else no use of redundant inclusion?
No centroid clustering, but cluster may have core subset.
'''

def sum2graph_(graph_, fd):  # sum node and link params into graph, derH in agg+ or player in sub+

    Graph_ = []
    for graph in graph_:  # seq Gs
        if graph[1] < G_aves[fd]:  # form graph if val>min only
            continue
        Graph = Cgraph()
        Link_ = []
        for i, G in enumerate(graph[0]):
            sum_box(Graph.box, G.box)
            for i in 0,1: sum_derH([Graph.derT[i], Graph.valt[i], Graph.rdnt[i]], [G.derT[i], G.valt[i],G.rdnt[i]],0)
            link_ = G.link_t[fd]
            Link_[:] = list(set(Link_ + link_))
            for j, derG in enumerate(link_):
                for i in 0,1: sum_derH([G.derT[i], G.valt[i],G.rdnt[i]], [derG.derT[i], derG.valt[i],derG.rdnt[i]], 0)
                sum_box(G.box, derG.node_[0].box if derG.node_[1] is G else derG.node_[1].box)
                Graph.nval += G.nval
            for i in 0,1:
                Graph.valt[1][i] += G.valt[1][i]; Graph.rdnt[1][i] += G.rdnt[1][i]
            Graph.node_ += [G]
        
        for derG in Link_:
            DerT = [[],[]]
            for i in 0,1: 
                sum_derH([DerT[i], Graph.valt[i], Graph.rdnt[i]], [derG.derT[i], derG.valt[i], derG.rdnt[i]], 0)  # sum unique links only
                Graph.derT[i] += DerT[i]  # concatenate
        Graph_ += [Graph]

    return Graph_

# old:
def op_parT(_graph, graph, fcomp, fneg=0):  # unpack aggH( subH( derH -> ptuples

    _parT, parT = _graph.parT, graph.parT

    if fcomp:
        dparT,valT,rdnT = comp_unpack(_parT, parT, rn=1)
        return dparT,valT,rdnT
    else:
        _valT, valT = _graph.valT, graph.valT
        _rdnT, rdnT = _graph.rdnT, graph.rdnT
        for i in 0,1:
            sum_unpack([_parT[i], _valT[i], _rdnT[i]], [parT[i], valT[i],rdnT[i]])

# same as comp|sum unpack?:
def op_ptuple(_ptuple, ptuple, fcomp, fd=0, fneg=0):  # may be ptuple, vertuple, or ext

    aveG = G_aves[fd]
    if fcomp:
        dtuple=CQ(n=_ptuple.n)  # + ptuple.n / 2: average n?
        rn = _ptuple.n/ptuple.n  # normalize param as param*rn for n-invariant ratio: _param/ param*rn = (_param/_n)/(param/n)
    _idx, d_didx, last_i, last_idx = 0,0,-1,-1

    for _i, _didx in enumerate(_ptuple.Q):  # i: index in Qd: select param set, idx: index in full param set
        _idx += _didx; idx = last_idx+1
        for i, didx in enumerate(ptuple.Q[last_i+1:]):  # start after last matching i and idx
            idx += didx
            if _idx == idx:
                _par = _ptuple.Qd[_i]; par = ptuple.Qd[_i+i]
                if fcomp:  # comp ptuple
                    if ptuple.Qm: val =_par+par if fd else _ptuple.Qm[_i]+ptuple.Qm[_i+i]
                    else:         val = aveG+1  # default comp for 0der pars
                    if val > aveG:
                        if isinstance(par,list):
                            if len(par)==4: m,d = comp_aangle(_par,par)
                            else: m,d = comp_angle(_par,par)
                        else: m,d = comp_par(_par, par*rn, aves[idx], finv = not i and not ptuple.Qm)
                            # finv=0 if 0der I
                        dtuple.Qm+=[m]; dtuple.Qd+=[d]; dtuple.Q+=[d_didx+_didx]
                        dtuple.valt[0]+=m; dtuple.valt[1]+=d  # no rdnt, rdn = m>d or d>m?)
                else:  # sum ptuple
                    D, d = _ptuple.Qd[_i], ptuple.Qd[_i+i]
                    if isinstance(d, list):  # angle or aangle
                        for j, (P,p) in enumerate(zip(D,d)): D[j] = P-p if fneg else P+p
                    else: _ptuple.Qd[i] += -d if fneg else d
                    if _ptuple.Qm:
                        mpar = ptuple.Qm[_i+i]; _ptuple.Qm[i] += -mpar if fneg else mpar
                last_i=i; last_idx=idx  # last matching i,idx
                break
            elif fcomp:
                if _idx < idx: d_didx+=didx  # no dpar per _par
            else: # insert par regardless of _idx < idx:
                _ptuple.Q.insert[idx, didx+d_didx]
                _ptuple.Q[idx+1] -= didx+d_didx  # reduce next didx
                _ptuple.Qd.insert[idx, ptuple.Qd[idx]]
                if _ptuple.Qm: _ptuple.Qm.insert[idx, ptuple.Qm[idx]]
                d_didx = 0
            if _idx < idx: break  # no par search beyond current index
            # else _idx > idx: keep searching
            idx += 1
        _idx += 1
    if fcomp: return dtuple

def comp_ext(_ext, ext, dsub):
    # comp ds:
    (_L,_S,_A), (L,S,A) = _ext, ext
    dL=_L-L; mL=ave-abs(dL); dS=_S/_L-S/L; mS=ave-abs(dS)
    if isinstance(A,list): mA, dA = comp_angle(_A,A)
    else:
        dA=_A-A; mA=ave-abs(dA)
    dsub.ext[0][:]= mL,mS,mA; dsub.ext[1][:]= dL,dS,dA
    dsub.valt[0] += mL+mS+mA; dsub.valt[1] += dL+dS+dA

def sum_ext(_ext, ext):
    _dL,_dS,_dA = _ext[1]; dL,dS,dA = ext[1]
    if isinstance(dA,list):
        _dA[0]+=dA[0]; _dA[1]+=dA[1]
    else: _dA+=dA
    _ext[1][:] = _dL+dL,_dS+dS, _dA
    if ext[0]:
        for i, _par, par in enumerate(zip(_ext[0],ext[0])):
            _ext[i] = _par+par

''' if n roots: 
sum_derH(Graph.uH[0][fd].derH,root.derH) or sum_G(Graph.uH[0][fd],root)? init if empty
sum_H(Graph.uH[1:], root.uH)  # root of Graph, init if empty
'''

def sum_box(Box, box):
    Y, X, Y0, Yn, X0, Xn = Box;  y, x, y0, yn, x0, xn = box
    Box[:] = [Y + y, X + x, min(X0, x0), max(Xn, xn), min(Y0, y0), max(Yn, yn)]

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
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, CQ):  # not proto-graph or removed
                            graph.alt_graph_ += [alt_graph]
                            alt_graph.alt_graph_ += [graph]
                            # bilateral assign
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            if graph.alt_graph_:
                graph.alt_derHs = CQ()  # players if fsub? der+: derHs[-1] += player, rng+: players[-1] = player?
                for alt_graph in graph.alt_graph_:
                    sum_pH(graph.alt_derHs, alt_graph.derHs)  # accum alt_graph_ params
                    graph.alt_rdn += len(set(graph.derHs.H[-1].node_).intersection(alt_graph.derHs.H[-1].node_))  # overlap

# draft:
def sub_recursion_eval(root, graph_):  # eval per fork, same as in comp_slice, still flat derH, add Valt to return?

    termt = [1,1]
    for graph in graph_:
        node_ = copy(graph.node_); sub_G_t = []
        fr = 0
        for fd in 0,1:
            if graph.valt[fd] > G_aves[fd] * graph.rdnt[fd] and len(graph.node_) > ave_nsub:
                graph.rdnt[fd] += 1  # estimate, no node.rdnt[fd] += 1?
                termt[fd] = 0; fr = 1
                sub_G_t += [sub_recursion(graph, node_, fd)]  # comp_der|rng in graph -> parLayer, sub_Gs
            else:
                sub_G_t += [node_]
                if isinstance(root, Cgraph):
                    root.fback_t[fd] += [[graph.derH, graph.valt, graph.rdnt]]  # we need fback_t here? Else they all will be flat
        if fr:
            graph.node_ = sub_G_t
    for fd in 0,1:
        if termt[fd] and root.fback_t[fd]:  # no lower layers in any graph
           feedback(root, fd)

def sub_recursion(graph, node_, fd):  # rng+: extend G_ per graph, der+: replace G_ with derG_, Valt=[0,0]?

    comp_G_(graph.node_, pri_G_=None, f1Q=1, fd=fd)  # cross-comp all Gs in rng
    sub_G_t = form_graph_(graph)  # cluster sub_graphs via link_t

    for i, sub_G_ in enumerate(sub_G_t):
        if sub_G_:  # and graph.rdn > ave_sub * graph.rdn:  # from sum2graph, not last-layer valt,rdnt?
            for sub_G in sub_G_: sub_G.roott[i] = graph
            sub_recursion_eval(graph, sub_G_)

    return sub_G_t  # for 4 nested forks in replaced P_?