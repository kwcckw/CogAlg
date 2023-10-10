import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from collections import deque, defaultdict
from .classes import Cgraph, CderG, CPP
from .filters import ave_L, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd, ave_dI, ave_G, ave_M, ave_Ma, ave_cluster
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_ptuple, sum_ptuple, sum_dertuple, comp_derH, matchF
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
This resembles a neuron, which has dendritic tree as input and axonal tree as output. 
But we have recursively structured param sets packed in each level of these trees, which don't exist in neurons.
Diagram: 
https://github.com/boris-kz/CogAlg/blob/76327f74240305545ce213a6c26d30e89e226b47/frame_2D_alg/Illustrations/generic%20graph.drawio.png
-
Clustering criterion is G.M|D, summed across >ave vars if selective comp (<ave vars are not compared, so they don't add costs).
Fork selection should be per var or co-derived der layer or agg level. 
There are concepts that include same matching vars: size, density, color, stability, etc, but in different combinations.
Weak value vars are combined into higher var, so derivation fork can be selected on different levels of param composition.
'''

def vectorize_root(blob, verbose):  # vectorization pipeline is 3 composition levels of cross-comp,clustering:

    edge = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering
    comp_P_(edge)  # vertical, lateral-overlap P cross-comp -> PP clustering
    # PP cross-comp -> discontinuous graph clustering:
    for fd in 0,1:
        node_ = edge.node_t[fd]  # always PP_t
        if edge.valt[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.rdnt[fd]:
            G_= []
            for PP in node_:  # convert CPPs to Cgraphs:
                derH, valt, rdnt, maxt = PP.derH, PP.valt, PP.rdnt, [0,0]  # init aggH is empty:
                for dderH in derH: dderH += [[0,0]]  # add maxt
                  
                G_ += [Cgraph( ptuple=PP.ptuple, derH=[derH,valt,rdnt,maxt],
                               L=PP.ptuple[-1], box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box))]
                for param in ("I","G","M","Ma", "angle", "L", "distance"):
                    G_[-1].valHt[param][fd] = 0  # update each param val and rdn here, so i guess we need additianal param of graph such as  G.I, G.G, G.M, and etc ?
                    G_[-1].rdnHt[param][fd] = 0
            node_ = G_
            edge.valHt[0][0] = edge.valt[0]; edge.rdnHt[0][0] = edge.rdnt[0]  # copy
            agg_recursion(None, edge, node_, fd=0)  # edge.node_t = graph_t, micro and macro recursive


def agg_recursion(rroot, root, G_, fd):  # compositional agg+|sub+ recursion in root graph, clustering G_

    root.valHt[fd] += [0]; root.rdnHt[fd] += [1]  # sum in form_graph_t feedback
    for param, ave_param in zip(("I","G","M","Ma", "angle", "L", "distance"), (ave_dI, ave_G, ave_M, ave_Ma, ave_dangle, ave_L, ave_distance)):
        comp_G_(G_, param, ave_param, fd)  # rng|der cross-comp all Gs, nD array? form link_H per G
    
        GG_t = form_graph_t(root, G_, param)  # eval sub+ and feedback per graph
    # agg+ xcomp-> form_graph_t loop sub)agg+, vs. comp_slice:
    # sub+ loop-> eval-> xcomp
        for GG_ in GG_t:  # comp_G_ eval: ave_m * len*rng - fixed cost, root update in form_t:
            if root.valHt[0][-1] * (len(GG_)-1)*root.rng > G_aves[fd] * root.rdnHt[0][-1]:
                agg_recursion(rroot, root, GG_, fd=0)  # 1st xcomp in GG_

        G_[:] = GG_t

def form_graph_t(root, G_, param):  # form mgraphs and dgraphs of same-root nodes

    graph_t = []
    for G in G_: G.root[param] = [None,None]  # replace with mcG_|dcG_ in segment_node_, replace with Cgraphs in sum2graph
    for fd in 0,1:
        Gt_ = sum_link_tree_(G_, fd)  # sum surround link values @ incr rng,decay
        graph_t += [segment_node_(root, Gt_, fd)]  # add alt_graphs?

    # below not updated
    # eval sub+, not in segment_node_: full roott must be replaced per node within recursion
    for fd, graph_ in enumerate(graph_t): # breadth-first for in-layer-only roots
        root.valHt[fd]+=[0]; root.rdnHt[fd]+=[1]  # remove if stays 0?

        for graph in graph_:  # external to agg+ vs internal in comp_slice sub+
            node_ = graph.node_t  # init flat
            if sum(graph.valHt[fd]) * (len(node_)-1)*root.rng > G_aves[fd] * sum(graph.rdnHt[fd]):  # eval fd comp_G_ in sub+
                agg_recursion(root, graph, node_, fd)  # replace node_ with node_t, recursive
            else:  # feedback after graph sub+:
                root.fback_t[fd] += [[graph.aggH, graph.valHt, graph.rdnHt, graph.maxHt]]
                root.valHt[fd][-1] += graph.valHt[fd][-1]  # last layer, or all new layers via feedback?
                root.rdnHt[fd][-1] += graph.rdnHt[fd][-1]  # merge forks in root fork
                root.maxHt[fd][-1] += graph.maxHt[fd][-1]
            i = sum(graph.valHt[0]) > sum(graph.valHt[1])
            root.rdnHt[i][-1] += 1  # add fork rdn to last layer, representing all layers after feedback

        if root.fback_t and root.fback_t[fd]:  # recursive feedback after all G_ sub+
            feedback(root, fd)  # update root.root.. aggH, valHt,rdnHt

    return graph_t  # root.node_t'node_ -> node_t: incr nested with each agg+?


def sum_link_tree_(node_,param, fd):  # sum surrounding link values to define connected nodes, with indirectly incr rng, to parallelize:
                               # link lower nodes via incr n of higher nodes, added till fully connected, n layers = med rng?
    ave = G_aves[fd]
    Gt_ = []
    for i, G in enumerate(node_):
        G.it[fd] = i  # used here and segment_node_
        Gt_ += [[G, G.valHt[fd][param][-1], G.rdnHt[fd][param][-1]]]  # init surround val,rdn
    # iterative eval rng expansion by summing decayed surround node Vals, while significant Val update:
    while True:
        DVal = 0
        for i, (_G,_Val,_Rdn) in enumerate(Gt_):
            if _Val < ave_cluster: continue
            Val, Rdn = 0, 0  # updated surround
            for link in _G.link_H[param][-1]:
                if link.valt[fd] < ave * link.rdnt[fd]: continue  # skip negative links
                G = link.G if link._G is _G else link._G
                if G not in node_: continue
                Gt = Gt_[G.it[fd]]
                Gval = Gt[1]; Grdn = Gt[2]
                try: decay = link.valt[fd]/link.maxt[fd]  # val rng incr per loop, per node?
                except: decay = 1  # /0
                Val += Gval * decay; Rdn += Grdn * decay  # link decay coef: m|d / max, base self/same
                # prune links by rng Val-ave*Rdn?
            Gt_[i][1] = Val; Gt_[i][2] = Rdn  # unilateral update, computed separately for _G
            DVal += abs(_Val-Val)  # _Val update / surround extension

        if DVal < ave:  # also if low Val?
            break
    return Gt_


def comp_G_(G_, param, ave_param, fd=0, oG_=None, fin=1):  # cross-comp in G_ if fin, else comp between G_ and other_G_, for comp_node_

    if not fd:  # cross-comp all Gs in extended rng, add proto-links regardless of prior links
        if param == "I":
            # 7 params set: I, G, M, Ma, angle, L, distance 
            new_link_layer = {"I":[], "G":[], "M":[], "Ma":[], "angle":[], "L":[], "distance":[]}
            for G in G_: G.link_H += [new_link_layer]  # add empty link layer, may remove if stays empty
            if oG_:
                for oG in oG_: oG.link_H += [deepcopy(new_link_layer)]
        # form new links:
        for i, G in enumerate(G_):
            if fin: _G_ = G_[i+1:]  # xcomp G_
            else:   _G_ = oG_       # xcomp G_,other_G_, also start @ i? not used yet
            for _G in _G_:
                if _G in G.compared_[param]: continue  # skip if previously compared
                
                # not sure below, add new param of I, G, M, Ma per graph ?
                if param == "I":
                    comp_eval = abs(_G.ptuple[0] - G.ptuple[0]) < ave_param
                elif param =="G":
                    comp_eval = abs(_G.ptuple[1] - G.ptuple[1]) < ave_param
                elif param =="M":
                    comp_eval = abs(_G.ptuple[2] - G.ptuple[2]) < ave_param
                elif param =="Ma":
                    comp_eval = abs(_G.ptuple[3] - G.ptuple[3]) < ave_param
                elif param == "L":
                    comp_eval = abs(getattr(_G, "L") -  getattr(G, "L")) < ave_param
                elif param == "angle":
                    mAngle, dAngle = comp_angle((_G.A[0],_G.A[1]), (G.A[0],G.A[1]))
                    comp_eval = mAngle < ave_param
                    
                elif param == "distance":
                    dy = _G.box[0] - G.box[0]; dx = _G.box[1] - G.box[1]
                    distance = np.hypot(dy, dx)  # Euclidean distance between centers of Gs
                    comp_eval = distance < ave_param  # close enough to compare
                
                if comp_eval:
                    # * ((sum(_G.valHt[fd]) + sum(G.valHt[fd])) / (2*sum(G_aves)))):  # comp rng *= rel value of comparands?
                    G.compared_[param] += [_G]; _G.compared_[param] += [G]
                    G.link_H[-1][param] += [CderG( G=G, _G=_G, S=distance, A=[dy,dx])]  # proto-links, in G only
    for G in G_:
        link_ = []
        for link in G.link_H[-1][param]:  # if fd: follow links, comp old derH, else follow proto-links, form new derH
            if fd and link.valt[1] < G_aves[1]*link.rdnt[1]: continue  # maybe weak after rdn incr?
            
            comp_G(link_,link, fd)
        G.link_H[-1][param] = link_

# draft
def comp_G(link_, link, fd):

    # for param set comp_G, it will be on certain param instead of whole ptuple and derH or aggH?
    pass