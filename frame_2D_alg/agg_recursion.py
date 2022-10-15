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
'''

ave_G = 6  # fixed costs per G
ave_Gm = 5  # for inclusion in graph
ave_Gd = 4
G_aves = [ave_Gm, ave_Gd]
ave_med = 3  # call cluster_node_layer
ave_rng = 3  # rng per combined val

class Cgraph(CPP):  # graph or generic PP of any composition

    plevels = list  # plevel_t[1]s is summed from alt_graph_, sub comp support, agg comp suppression?
    fds = list  # prior forks in plevels, then player fds in plevel
    valt = lambda: [0, 0]  # mval, dval from cis+alt forks
    nvalt = lambda: [0, 0]  # from neg open links
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs; no alt_rdn: valt representation in alt_PP_ valts?
    rng = lambda: 1  # not for alt_graphs
    link_ = list  # all evaluated external graph links, nested in link_layers? open links replace alt_node_
    node_ = list  # graph elements, root of layers and levels:
    rlayers = list  # | mlayers, top-down
    dlayers = list  # | alayers
    mlevels = list  # agg_PPs ) agg_PPPs ) agg_PPPPs.., bottom-up
    dlevels = list
    roott = lambda: [None, None]  # higher-order segG or graph
    alt_graph_ = list


def agg_recursion(root, G_, fseg):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

    ivalt = root.valt
    mgraph_, dgraph_ = form_graph_(root, G_)  # PP cross-comp and clustering

    # intra graph:
    if root.valt[0] > ave_sub * root.rdn:
        sub_rlayers, rvalt = sub_recursion_g(mgraph_, root.valt, fd=0)  # subdivide graph.node_ by der+|rng+, accum root.valt
        root.valt[0] += sum(rvalt); root.rlayers = sub_rlayers
    if root.valt[1] > ave_sub * root.rdn:
        sub_dlayers, dvalt = sub_recursion_g(dgraph_, root.valt, fd=1)
        root.valt[1] += sum(dvalt); root.dlayers = sub_dlayers

    # cross graph:
    root.mlevels += mgraph_; root.dlevels += dgraph_
    for fd, graph_ in enumerate([mgraph_, dgraph_]):
        adj_val = root.valt[fd] - ivalt[fd]
        # recursion if adjusted val:
        if (adj_val > G_aves[fd] * ave_agg * (root.rdn + 1)) and len(graph_) > ave_nsub:
            root.rdn += 1  # estimate
            agg_recursion(root, graph_, fseg=fseg)  # cross-comp graphs


def form_graph_(root, G_):  # G is potential node graph, in higher-order GG graph

    for G in G_:  # initialize mgraph, dgraph as roott per G, for comp_G_
        for i in 0,1:
            graph = [[G], [], [0,0]]  # proto-GG: [node_, meds_, valt]
            G.roott[i] = graph

    comp_G_(G_)  # cross-comp all graphs within rng, graphs may be segs
    mgraph_, dgraph_ = [],[]  # initialize graphs with >0 positive links in graph roots:
    for G in G_:
        if len(G.roott[0][0])>1: mgraph_ += [G.roott[0]]  # root = [node_, valt] for cluster_node_layer eval, + link_nvalt?
        if len(G.roott[1][0])>1: dgraph_ += [G.roott[1]]

    for fd, graph_ in enumerate([mgraph_, dgraph_]):  # evaluate intermediate nodes to delete or merge graphs:
        regraph_ = []
        while graph_:
            graph = graph_.pop(0)
            eval_med_layer(graph_= graph_, graph=graph, fd=fd)
            if graph[2][fd] > ave_agg: regraph_ += [graph]  # graph reformed by merges and removes above

        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
            plevels = deepcopy(graph_[0].plevels)
            for graph in graph_[1:]:
                sum_plevels(plevels, graph.plevels)  # each plevel is (caTree, valt)
            root.plevels = plevels

    return mgraph_, dgraph_


def comp_G_(G_):  # cross-comp Gs (patterns of patterns): Gs, derGs, or segs inside PP

    for i, _G in enumerate(G_):

        for G in G_[i+1:]:  # compare each G to other Gs in rng, bilateral link assign
            if G in [node for link in _G.link_ for node in link.node_]:  # G,_G was compared in prior rng+, add frng to skip?
                continue
            dx = (_G.xn-_G.x0)/2 - (G.xn-G.x0)/2; dy = (_G.yn-_G.y0)/2 - (G.yn-G.y0)/2
            distance = np.hypot(dy, dx)  # Euclidean distance between centroids, max depends on combined value:
            if distance <= ave_rng * ((sum(_G.valt)+sum(G.valt)) / (2*sum(G_aves))):

                plevel, valt = comp_plevels(_G.plevels, G.plevels, _G.fds, G.fds)
                valt = [valt[0] - ave_Gm, valt[1] - ave_Gd]  # or already normalized, *= link rdn?
                # separate valt param anyway?
                derG = Cgraph(
                    plevels=plevel, x0=min(_G.x0,G.x0), xn=max(_G.xn,G.xn), y0=min(_G.y0,G.y0), yn=max(_G.yn,G.yn), valt=valt, node_=[_G,G])
                _G.link_ += [derG]; G.link_ += [derG]  # any val
                for fd in 0,1:
                    if valt[fd] > 0:  # alt fork is redundant, no support?
                        for node, (graph, meds_, gvalt) in zip([_G, G], [G.roott[fd], _G.roott[fd]]):  # bilateral inclusion
                            if node not in graph:
                                graph += [node]
                                meds_ += [[derG.node_[0] if derG.node_[1] is node else derG.node_[1] for derG in node.link_]]  # immediate links
                                gvalt[0] += node.valt[0]; gvalt[1] += node.valt[1]


def eval_med_layer(graph_, graph, fd):   # recursive eval of reciprocal links from increasingly mediated nodes

    node_, meds_, valt = graph
    save_node_, save_meds_ = [], []
    adj_Val = 0  # adjust connect val in graph

    for G, med_node_ in zip(node_, meds_):  # G: node or sub-graph
        mmed_node_ = []  # __Gs that mediate between Gs and _Gs
        for _G in med_node_:
            for derG in _G.link_:
                if derG not in G.link_:  # link_ includes all unique evaluated mediated links, flat or in layers?
                    # med_PP.link_:
                    med_link_ = derG.node_[0].link_ if derG.node_[0] is not _G else derG.node_[1].link_
                    for _derG in med_link_:
                        if G in _derG.node_ and _derG not in G.link_:  # __G mediates between _G and G
                            G.link_ += [_derG]
                            adj_val = _derG.valt[fd] - ave_agg  # or increase ave per mediation depth
                            # adjust nodes:
                            G.valt[fd] += adj_val; _G.valt[fd] += adj_val
                            valt[fd] += adj_val; _G.roott[fd][2][fd] += adj_val  # root is not graph yet
                            __G = _derG.node_[0] if _derG.node_[0] is not _G else _derG.node_[1]
                            if __G not in mmed_node_:  # not saved via prior _G
                                mmed_node_ += [__G]
                                adj_Val += adj_val
        if G.valt[fd]>0:
            # G remains in graph
            save_node_ += [G]; save_meds_ += [mmed_node_]  # mmed_node_ may be empty

    for G, mmed_ in zip(save_node_, save_meds_):  # eval graph merge after adjusting graph by mediating node layer
        add_mmed_= []
        for _G in mmed_:
            _graph = _G.roott[fd]
            if _graph in graph_ and _graph is not graph:  # was not removed or merged via prior _G
                _node_, _meds_, _valt = _graph
                for _node, _meds in zip(_node_, _meds_):  # merge graphs, ignore _med_? add direct links:
                    for derG in _node.link_:
                        __G = derG.node_[0] if derG.node_[0] is not _G else derG.node_[1]
                        if __G not in add_mmed_ + mmed_:  # not saved via prior _G
                            add_mmed_ += [__G]
                            adj_Val += derG.valt[fd] - ave_agg
                valt[fd] += _valt[fd]
                graph_.remove(_graph)
        mmed_ += add_mmed_

    graph[:] = [save_node_,save_meds_,valt]
    if adj_Val > ave_med:  # positive adj_Val from eval mmed_
        eval_med_layer(graph_, graph, fd)  # eval next med layer in reformed graph


def sub_recursion_g(graph_, fseg, fd):  # rng+: extend G_ per graph, der+: replace G_ with derG_

    comb_layers_t = [[],[]]
    sub_valt = [0,0]
    for graph in graph_:
        if fd:
            node_ = []  # positive links within graph
            for node in graph.node_:
                for link in node.link_:
                    if link.valt[0]>0 and link not in node_:
                        node_ += [link]
        else: node_ = graph.node_

        # rng+|der+ if top player valt[fd], for plevels[:-1]| players[-1][fd=1]:  (graph.valt eval for agg+ only)
        if graph.plevels[-1][-1][2][fd] > G_aves[fd] and len(node_) > ave_nsub:

            sub_mgraph_, sub_dgraph_ = form_graph_(graph, node_)  # cross-comp and clustering cycle
            # rng+:
            if graph.valt[0] > ave_sub * graph.rdn:  # >cost of calling sub_recursion and looping:
                sub_rlayers, valt = sub_recursion_g(sub_mgraph_, graph.valt, fd=0)
                rvalt = sum(valt); graph.valt[0] += rvalt; sub_valt[0] += rvalt  # not sure
                graph.rlayers = [sub_mgraph_] + [sub_rlayers]
            # der+:
            if graph.valt[1] > ave_sub * graph.rdn:
                sub_dlayers, valt = sub_recursion_g(sub_dgraph_, graph.valt, fd=1)
                dvalt = sum(valt); graph.valt[1] += dvalt; sub_valt[1] += dvalt
                graph.dlayers = [sub_dgraph_] + [sub_dlayers]

            for comb_layers, graph_layers in zip(comb_layers_t, [graph.rlayers, graph.dlayers]):
                for i, (comb_layer, graph_layer) in enumerate(zip_longest(comb_layers, graph_layers, fillvalue=[])):
                    if graph_layer:
                        if i > len(comb_layers) - 1:
                            comb_layers += [graph_layer]  # add new r|d layer
                        else:
                            comb_layers[i] += graph_layer  # splice r|d PP layer into existing layer

    return comb_layers_t, sub_valt


def sum2graph_(G_, fd):  # sum node and link params into graph
    # fsub if from sub_recursion: add players vs. plevels?

    graph_ = []  # new graph_
    for G in G_:
        node_, meds_, valt = G
        node = node_[0]  # init graph with 1st node:
        graph = Cgraph( plevels=deepcopy(node.plevels), fds=deepcopy(node.fds), valt=node.valt,
                        x0=node.x0, xn=node.xn, y0=node.y0, yn=node.yn, node_ = node_, meds_ = meds_)

        derG = node.link_[0]  # init new_plevel with 1st derG:
        new_plevel = derG.plevels[fd]; derG.roott[fd] = graph; valt[0] += derG.valt[0]; valt[1] += derG.valt[1]
        for derG in node.link_[1:]:
            sum_derG(new_plevel, derG.plevels[fd])  # accum derG in new plevel
            valt[0] += derG.valt[0]; valt[1] += derG.valt[1]
            derG.roott[fd] = graph
        for node in node_[1:]:
            graph.valt[0] += node.valt[0]; graph.valt[1] += node.valt[1]
            graph.x0=min(graph.x0, node.x0); graph.xn=max(graph.xn, node.xn); graph.y0=min(graph.y0, node.y0); graph.yn=max(graph.yn, node.yn)
            # accum params:
            sum_plevels(graph.plevels, node.plevels)
            for derG in node.link_:
                sum_derG(new_plevel, derG.plevels[fd])  # accum derG, add to graph when complete
                valt[0] += derG.valt[0]; valt[1] += derG.valt[1]
                derG.roott[fd] = graph
                # link_ = [derG]?
        graph_ += [graph]
        graph.plevels += [new_plevel]

    for graph in graph_:  # 2nd pass: accum alt_graph_ params
        Alt_plevels = []
        for node in graph.node_:
            for derG in node.link_:
                for G in derG.node_:
                    if G not in graph.node_:  # alt graphs are roots of not-in-graph G in derG.node_
                        alt_graph = G.roott[fd]
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, Cgraph):  # not proto-graph
                            if Alt_plevels: sum_plevels(Alt_plevels, alt_graph.plevels)
                            else:           Alt_plevels = deepcopy(alt_graph.plevels)
                            graph.alt_graph_ += [alt_graph]
        if graph.alt_graph_:
            graph.alt_graph_ += [Alt_plevels]  # temporary storage

    for graph in graph_:  # 3rd pass: add alt fork to each graph plevel, separate to fix nesting in 2nd pass
        if graph.alt_graph_:
            Alt_plevels = graph.alt_graph_.pop()
        else: Alt_plevels = []

        ca_splice(graph.plevels, Alt_plevels)  # recursive unpack and splice alt forks

    return graph_

def ca_splice_deep(cQue, aQue):  # Que is plevels or players, unpack and splice alt forks in two calls
    fplayers=0

    for (cTree, cvalt), (aTree, avalt) in zip(cQue, aQue):
        cvalt[0] += avalt[0]; avalt[1] += avalt[1]
        cTree += cTree
        for cQt, aQt in zip(cTree, aTree):
            if cQt:
                if len(cQt) == 2: cQ, cvalt = cQt; fplayers = 1  # Q is players
                else:             cQ, cfds, cvalt = cQt  # Q is ptuples
            else: cQ = [], cvalt = [0, 0]
            if aQt:
                if len(aQt) == 2: aQ, avalt = aQt
                else:             aQ, afds, avalt = cQt
            else: aQ = [], avalt = [0, 0]

            cvalt[0] += avalt[0]; avalt[1] += avalt[1]
            cQ += aQ

            if fplayers:
                ca_splice(cQ, aQ)


def ca_splice(cQue, aQue):  # Que is plevels or players, unpack and splice alt forks in two calls

    for (cTree, cvalt), (aTree, avalt) in zip(cQue, aQue):
        cvalt[0] += avalt[0]; avalt[1] += avalt[1]
        cTree += aTree  # Tree is flat list,
        # aQs remain packed in aTree?

# need to revise:
''' 
plevels ( caTree ( players ( caTree ( ptuples )))
'''
def comp_plevels(_plevels, plevels, _fds, fds):  # each plevel is caTree|caT: binary cis,alt tree with players as leaves

    mplevel, dplevel = [],[]  # fd plevels, each cis+alt, same as new_caT
    Valt = [0,0]  # each cis+alt
    iVal = ave_G  # to start loop:

    for (_caTree, cvalt), (caTree, cvalt), _fd, fd in zip(reversed(_plevels), reversed(plevels), _fds, fds):
        if iVal < ave_G:  # top-down, comp if higher plevels match, same agg+
            break
        oplayers, alt_oplayers = [], []; pvalt = [0,0]
        for _players, players, _fd, fd in zip(_caTree, caTree, _fds, fds):  # bottom-up der+
            if _fd == fd:
                if _players and _players:  # skip alts if empty
                    mplayert, dplayert = comp_players(_players, players)
                    oplayers     += [mplayert]; pvalt[0] += mplayert[1]
                    alt_oplayers += [dplayert]; pvalt[1] += dplayert[1]
            else:
                break  # comp same fds
        mplevel += [oplayers]; dplevel += [alt_oplayers]
        for i in 0,1: Valt[i] += pvalt[i]  # new plevel is m,d pair of candidate plevels
        iVal = sum(Valt)  # after 1st loop

    # if there's no bracket, it wil be returned as tuple - ([mplevel, dplevel], Valt)
    return [[mplevel, dplevel], Valt]  # always single new plevel


def comp_players(_playerst, playerst):  # unpack and compare der layers, if any from der+

    mplayer, dplayer = [], []  # flat lists of ptuples, nesting decoded by mapping to lower levels
    mval, dval = 0, 0  # new, the old ones in valt for sum2graph
    _players, _fds, _valt = _playerst
    players, fds, valt = playerst

    for (_player_caTree, valt), (player_caTree, valt), _fd, fd in zip_longest(_players, players, _fds, fds, fillvalue=[]):
        mplayert, dplayert = [[],[]], [[],[]]
        if _fd==fd:
            for i, (_player, player) in enumerate(zip(_player_caTree, player_caTree)):  # always 2 elements here right? for playert
                for _ptuple, ptuple in zip(_player, player):
                    mtuple, dtuple = comp_ptuple(_ptuple, ptuple)
                    mplayert[i] += [mtuple]; mval += mtuple.val
                    dplayert[i] += [dtuple]; dval += dtuple.val
        else:
            break  # comp same fds
        mplayer.append(mplayert)
        dplayer.append(dplayert)

    # add additional bracket because players in nested, player is not
    return [[mplayer],mval], [[dplayer],dval]  # as in comp_plevels, single new lplayer, no fds till sum2graph


def sum_plevels(pLevels, plevels):

    for CaTreet, caTreet in zip_longest(pLevels, plevels, fillvalue=[]):  # loop top-down for selective comp depth, same agg+?
        if caTreet:  # plevel is not empty
            if CaTreet:
                CaTree, valt = CaTreet; caTree, valt = caTreet
                for Playerst, playerst in zip(CaTree, caTree):  
                    for Playerst, playerst in zip_longest(Playerst, playerst, fillvalue=[]):
                        if len(Playerst) == 3:
                            sum_playerst_fd(Playerst, playerst)
                        else:
                            sum_playerst(Playerst, playerst)
            else:  # add new plevel
                pLevels.append(deepcopy(caTreet))

# draft
def sum_derG(pLevelt, plevelt):

    for FdQ, fdQ in zip_longest(pLevelt[:-1], plevelt[:-1], fillvalue=[]):  # alts may be empty
        if fdQ:
            if FdQ:
                for Playerst, playerst in zip(FdQ, fdQ):
                    sum_playerst(Playerst, playerst)
            else:
                FdQ[:] = deepcopy(fdQ)

# not updated
def sum_playerst(Playerst, playerst):  # accum layers while same fds

    Players, Valt = Playerst
    players, valt = playerst
    for i, (pLayer, player) in enumerate( zip_longest(Players, players, fillvalue=[])):
        if player:
            if pLayer: sum_player(pLayer, player, fneg=0)
            else:      Players += [player]


# not updated:
def sum_playerst_fd(pLayerst, playerst):  # accum layers while same fds

    pLayers, Fds, Valt = pLayerst
    players, fds, valt = playerst
    fbreak = 0

    for i, (pLayer, player, Fd, fd) in enumerate( zip_longest(pLayers, players, Fds, fds, fillvalue=[])):
        if Fd==fd:
            if player:
                if pLayer: sum_player(pLayer, player, fneg=0)
                else:
                    pLayers+=[player]
                    # need to sum vals per player:
                    # Val, val in zip(Valt, valt): Val+=val?
        else:
            fbreak = 1
            break
    Fds[:] = Fds[:i + 1 - fbreak]


# not revised, this is an alternative to form_graph, but may not be accurate enough to cluster:
def comp_centroid(PPP_):  # comp PP to average PP in PPP, sum >ave PPs into new centroid, recursion while update>ave

    update_val = 0  # update val, terminate recursion if low

    for PPP in PPP_:
        PPP_valt = [0 ,0]  # new total, may delete PPP
        PPP_rdn = 0  # rdn of PPs to cPPs in other PPPs
        PPP_players_t = [[], []]
        DerPP = CderG(player=[[], []])  # summed across PP_:
        Valt = [0, 0]  # mval, dval

        for i, (PP, _, fint) in enumerate(PPP.PP_):  # comp PP to PPP centroid, derPP is replaced, use comp_plevels?
            Mplayer, Dplayer = [],[]
            # both PP core and edge are compared to PPP core, results are summed or concatenated:
            for fd in 0, 1:
                if PP.players_t[fd]:  # PP.players_t[1] may be empty
                    mplayer, dplayer = comp_players(PPP.players_t[0], PP.players_t[fd], PPP.fds, PP.fds)  # params norm in comp_ptuple
                    player_t = [Mplayer + mplayer, Dplayer + dplayer]
                    valt = [sum([mtuple.val for mtuple in mplayer]), sum([dtuple.val for dtuple in dplayer])]
                    Valt[0] += valt[0]; Valt[1] += valt[1]  # accumulate mval and dval
                    # accum DerPP:
                    for Ptuple, ptuple in zip_longest(DerPP.player_t[fd], player_t[fd], fillvalue=[]):
                        if ptuple:
                            if not Ptuple: DerPP.player_t[fd].append(ptuple)  # pack new layer
                            else:          sum_players([[Ptuple]], [[ptuple]])
                    DerPP.valt[fd] += valt[fd]

            # compute rdn:
            cPP_ = PP.cPP_  # sort by derPP value:
            cPP_ = sorted(cPP_, key=lambda cPP: sum(cPP[1].valt), reverse=True)
            rdn = 1
            fint = [0, 0]
            for fd in 0, 1:  # sum players per fork
                for (cPP, CderG, cfint) in cPP_:
                    if valt[fd] > PP_aves[fd] and PP.players_t[fd]:
                        fint[fd] = 1  # PPs match, sum derPP in both PPP and _PPP:
                        sum_players(PPP.players_t[fd], PP.players_t[fd])
                        sum_players(PPP.players_t[fd], PP.players_t[fd])  # all PP.players in each PPP.players

                    if CderG.valt[fd] > Valt[fd]:  # cPP is instance of PP
                        if cfint[fd]: PPP_rdn += 1  # n of cPPs redundant to PP, if included and >val
                    else:
                        break  # cPP_ is sorted by value

            fnegm = Valt[0] < PP_aves[0] * rdn;  fnegd = Valt[1] < PP_aves[1] * rdn  # rdn per PP
            for fd, fneg, in zip([0, 1], [fnegm, fnegd]):

                if (fneg and fint[fd]) or (not fneg and not fint[fd]):  # re-clustering: exclude included or include excluded PP
                    PPP.PP_[i][2][fd] = 1 -  PPP.PP_[i][2][fd]  # reverse 1-0 or 0-1
                    update_val += abs(Valt[fd])  # or sum abs mparams?
                if not fneg:
                    PPP_valt[fd] += Valt[fd]
                    PPP_rdn += 1  # not sure
                if fint[fd]:
                    # include PP in PPP:
                    if PPP_players_t[fd]: sum_players(PPP_players_t[fd], PP.players_t[fd])
                    else: PPP_players_t[fd] = copy(PP.players_t[fd])  # initialization is simpler
                    # not revised:
                    PPP.PP_[i][1] = derPP   # no derPP now?
                    for i, cPPt in enumerate(PP.cPP_):
                        cPPP = cPPt[0].root
                        for j, PPt in enumerate(cPPP.cPP_):  # get PPP and replace their derPP
                            if PPt[0] is PP:
                                cPPP.cPP_[j][1] = derPP
                        if cPPt[0] is PP: # replace cPP's derPP
                            PPP.cPP_[i][1] = derPP
                PPP.valt[fd] = PPP_valt[fd]

        if PPP_players_t: PPP.players_t = PPP_players_t

        # not revised:
        if PPP_val < PP_aves[fPd] * PPP_rdn:  # ave rdn-adjusted value per cost of PPP

            update_val += abs(PPP_val)  # or sum abs mparams?
            PPP_.remove(PPP)  # PPPs are hugely redundant, need to be pruned

            for (PP, derPP, fin) in PPP.PP_:  # remove refs to local copy of PP in other PPPs
                for (cPP, _, _) in PP.cPP_:
                    for i, (ccPP, _, _) in enumerate(cPP.cPP_):  # ref of ref
                        if ccPP is PP:
                            cPP.cPP_.pop(i)  # remove ccPP tuple
                            break

    if update_val > sum(PP_aves):
        comp_centroid(PPP_)  # recursion while min update value

    return PPP_