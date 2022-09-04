import sys
import numpy as np
from itertools import zip_longest
from copy import deepcopy, copy
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
import math as math
from comp_slice import *

'''
Blob edges may be represented by higher-composition PPPs, etc., if top param-layer match,
in combination with spliced lower-composition PPs, etc, if only lower param-layers match.
This may form closed edge patterns around flat blobs, which defines stable objects.   
'''

class CderPP(ClusterStructure):  # tuple of derivatives in PP uplink_ or downlink_, PP can also be PPP, etc.

    player_t = lambda: [[], []]  # lists of ptuples in current derivation layer per fork
    valt = lambda: [0,0]  # tuple of mval, dval
    # 5 below are not needed?
    box = list  # 2nd level, stay in PP?
    rdn = int  # mrdn, + uprdn if branch overlap?
    _PP = object  # prior comparand  or always in PP_?
    PP = object  # next comparand
    root = lambda: None  # PP, not segment in sub_recursion
    # not needed?
    uplink_layers = lambda: [[], []]  # init a layer of dderPs and a layer of match_dderPs
    downlink_layers = lambda: [[], []]
    # from comp_dx
    fdx = NoneType

class CgPP(CPP, CderPP):  # generic PP, of any composition

    players_t = lambda: [[], []]  # mPlayers, dPlayers, max n ptuples / layer = n ptuples in all lower layers: 1, 1, 2, 4, 8...
    valt = lambda: [0, 0]  # mval, dval
    nvalt = lambda: [0, 0]  # neg links
    fds = list  # prior fork sequence, map to mplayer or dplayer in lower (not parallel) player, none for players[0]
    rng = lambda: 1  # rng starts with 1
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs
    Rdn = int  # for accumulation only
    nP = int  # len 2D derP__ in levels[0][fPd]?  ly = len(derP__), also x, y?
    uplink_layers = lambda: [[], []]  # likely not needed
    downlink_layers = lambda: [[], []]
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    box = list  # x0, xn, y0, yn
    mask__ = bool
    gPP_ = list  # (PP|gPP,derPP,fin)s, common root of layers and levels
    cPP_ = list  # co-refs in other PPPs
    rlayers = list  # | mlayers
    dlayers = list  # | alayers
    mlevels = list  # agg_PPs ) agg_PPPs ) agg_PPPPs..
    dlevels = list
    roott = lambda: [None, None]  # lambda: CgPP()  # higher-order segP or PPP


def agg_recursion(root, PP_, rng, fseg=0):  # compositional recursion per blob.Plevel; P, PP, PPP are relative to each other

    PPP_ = comp_PP_(copy(PP_), rng, fseg)  # cross-comp all PPs (which may be segs) within rng, same PPP_ for both forks
    graph_ = form_graph(PPP_, rng)  # if top level miss, lower levels match: splice PPs vs form PPPs

    # intra-graph:
    if sum(root.valt) > ave_agg * root.rdn:
        # draft:
        sub_rlayers, rvalt = sub_recursion_agg(graph_, root.valt, fd=0)  # re-form PPP.PP_ by der+ if PPP.fPd else rng+, accum root.valt
        root.valt[0] += sum(rvalt); root.rlayers = sub_rlayers
        sub_dlayers, dvalt = sub_recursion_agg(graph_, root.valt, fd=1)
        root.valt[1] += sum(dvalt); root.dlayers = sub_dlayers
    # cross-graph:
    val = sum(root.valt)
    if (val > ave_agg * (root.rdn/len(graph_))) and len(graph_) > ave_nsub:  # not sure about rdn norm
        root.rdn += 1  # estimate, replace by actual after agg_recursion?
        root.mlevels += graph_  # bottom-up
        root.dlevels += []  # none for now
        agg_recursion(root, graph_, rng = val/ave_agg, fseg=0)  # cross-comp graphs


def comp_PP_(PP_, rng, fseg):  # 1st cross-comp, PPs may be segs inside a PP

    PPP_ = []
    iPPP_ = [copy_P(PP, iPtype=4) for PP in PP_]

    while PP_:  # compare _PP to all other PPs within rng
        _PP, _PPP = PP_.pop(), iPPP_.pop()
        _fid = _PP.fds[-1]
        _PP.roott[_fid] = _PPP

        for PPP, PP in zip(iPPP_, PP_):  # add selection by dy<rng if incremental y? accum derPPs in PPPs
            fid = PP.fds[-1]
            area = PP.players_t[fid][0][0].L; _area = _PP.players_t[_fid][0][0].L
            dx = ((_PP.xn - _PP.x0) / 2) / _area - ((PP.xn - PP.x0) / 2) / area
            dy = _PP.y / _area - PP.y / area
            distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids

            val = sum(_PP.valt) / (ave_mPP+ave_dPP)  # combined PP eval
            if distance * val <= rng:
                # comp PPs:
                mplayer, dplayer = comp_players(_PP.players_t[_fid], PP.players_t[fid], _PP.fds, PP.fds)
                valt = [sum([mtuple.val for mtuple in mplayer]), sum([dtuple.val for dtuple in dplayer])]
                derPP = CderPP(player=[mplayer, dplayer], valt=valt)  # derPP is single-layer
                fint = []
                for fd in 0,1:  # sum players per fork
                    if valt[fd] > PP_aves[fd]:
                        fin = 1  # PPs match, sum derPP in both PPP and _PPP:
                        sum_players(_PPP.players_t[fd], PP.players_t[fd])
                        sum_players(PPP.players_t[fd], PP.players_t[fd])  # all PP.players in each PPP.players
                    else:
                        fin = 0
                    fint += [fin]
                _PPP.gPP_ += [[PP, derPP, fint]]
                _PP.cPP_ += [[PP, derPP, [1,1]]]  # rdn refs, initial fins=1, derPP is reversed
                PPP.gPP_ += [[_PP, derPP, fint]]
                PP.cPP_ += [[_PP, derPP, [1,1]]]  # bilateral assign to eval in centroid clustering, derPP is reversed
                '''
                if derPP.match params[-1]: form PPP
                elif derPP.match params[:-1]: splice PPs and their segs? 
                '''
        PPP_.append(_PPP)
    return PPP_


def form_graph(PPP_, rng):  # cluster PPPs by match of mutual connections, which is summed across shared nodes and their roots

    graph_= []
    for PPP in PPP_:  # initialize graph for each PPP:
        graph=CgPP(gPP_=[PPP], fds = copy(PPP.fds), rng=rng, players_t = deepcopy(PPP.players_t), valt=PPP.valt)  # graph.gPP_: connected PPPs
        PPP.roott[0]=graph  # set root of core part as graph
        graph_+=[graph]
    merged_graph_ = []
    while graph_:
        graph = graph_.pop(0)
        eval_ref_layer(graph_=graph_, graph=graph, PPP_=graph.gPP_, shared_M=0)
        merged_graph_ += [graph]  # graphs merge (split?) in eval_ref_layer, each layer is intermediate PPPs
        # add|remove individual nodes+links: PPPs, not the links between PPs in PPP?

    return merged_graph_

# draft
def eval_ref_layer(graph_, graph, PPP_, shared_M):  # recursive eval of increasingly mediated nodes (graph_, graph, shared_M)?

    for PPP in PPP_:
        for (PP, derPP, fint) in PPP.gPP_:
            shared_M += derPP.valt[0]  # accum shared_M across mediating node layers

            for (_PP, _derPP, _fint) in PP.roott[0].gPP_:  # _PP.PPP / PP.PPP reciprocal refs:
                if _PP is PP:  # mutual connection
                    shared_M += _derPP.valt[0] - ave_agg  # eval graph inclusion by match to mediating gPP
                    # * rdn: sum PP_ rdns / len PP_ + cross-PPP overlap rate + cross-graph overlap rate?
                    if shared_M > 0:
                        _graph = _PP.roott[0].roott[0]  # PP.PPP.graph[m]: two nesting layers above PP
                        if _graph is not graph:
                            # merge graphs:
                            for fd in 0, 1:
                                if _graph.players_t[fd]:
                                    sum_players(graph.players_t[fd], _graph.players_t[fd])
                                    graph.valt[fd] += _graph.valt[fd]
                            for rPP in _graph.gPP_: rPP.roott[0] = graph  # update graph reference
                            graph.gPP_ += _graph.gPP_
                            graph_.remove(_graph)

                            # recursive search of increasingly intermediated refs for mutual connections
                            for __PP in _PP.gPP_:
                                for (_root_PP,_,_) in __PP.root.gPP_:
                                    _root_PPP = _root_PP.roott[0]
                                    if _root_PPP.roott[0] is not graph:
                                        eval_ref_layer(graph_, graph, _root_PPP.roott[0].gPP_, shared_M)


# draft:
def sub_recursion_agg(graph_, fseg, fd):  # rng+: extend PP_ per PPP, der+: replace PP with derPP in PPt

    comb_layers = []
    sub_valt = [0,0]

    for graph in graph_:
        if graph.valt[fd] > PP_aves[fd] and len(graph.gPP_) > ave_nsub:

            sub_PPP_ = comp_PP_(graph.gPP_, fseg)  # cross-comp all PPs within rng, same PPP_ for both forks, add fseg?
            sub_graph_ = form_graph(sub_PPP_, graph.rng)  # if top level miss, lower levels match: splice PPs vs form PPPs

            sub_rlayers, valt = sub_recursion_agg(sub_graph_, graph.valt, fd=0)  # re-form PPP.PP_ by der+ if PPP.fPd else rng+, accum root.valt
            rvalt = sum(valt); graph.valt[0] += rvalt; sub_valt[0] += rvalt  # not sure
            sub_dlayers, valt = sub_recursion_agg(sub_graph_, graph.valt, fd=1)
            dvalt = sum(valt); graph.valt[1] += dvalt; sub_valt[1] += dvalt

            for i, (comb_layer, graph_layer) in enumerate(zip_longest(comb_layers, graph.dlayers if fd else graph.rlayers, fillvalue=[])):
                if graph_layer:
                    if i > len(comb_layers) - 1:
                        comb_layers += [graph_layer]  # add new r|d layer
                    else:
                        comb_layers[i] += graph_layer  # splice r|d PP layer into existing layer

    return comb_layers, sub_valt


# not fully revised, this is an alternative to form_graph, but may not be accurate enough to cluster:
def comp_centroid(PPP_):  # comp PP to average PP in PPP, sum >ave PPs into new centroid, recursion while update>ave

    update_val = 0  # update val, terminate recursion if low

    for PPP in PPP_:
        PPP_valt = [0 ,0]  # new total, may delete PPP
        PPP_rdn = 0  # rdn of PPs to cPPs in other PPPs
        PPP_players_t = [[], []]
        DerPP = CderPP(player=[[], []])  # summed across PP_:
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
                for (cPP, cderPP, cfint) in cPP_:
#                    if valt[fd] > PP_aves[fd] and PP.players_t[fd]:
                    if PP.players_t[fd]:
                        fint[fd] = 1  # PPs match, sum derPP in both PPP and _PPP:
                        sum_players(PPP.players_t[fd], PP.players_t[fd])
                        sum_players(PPP.players_t[fd], PP.players_t[fd])  # all PP.players in each PPP.players

                    if cderPP.valt[fd] > Valt[fd]:  # cPP is instance of PP
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


# for deeper agg_recursion:
def comp_levels(_levels, levels, der_levels, fsubder=0):  # only for agg_recursion, each param layer may consist of sub_layers

    # recursive unpack of nested param layers, each layer is ptuple pair_layers if from der+
    der_levels += [comp_players(_levels[0], levels[0])]

    # recursive unpack of deeper layers, nested in 3rd and higher layers, if any from agg+, down to nested tuple pairs
    for _level, level in zip(_levels[1:], levels[1:]):  # level = deeper sub_levels, stop if none
        der_levels += [comp_levels(_level, level, der_levels=[], fsubder=fsubder)]

    return der_levels  # possibly nested param layers


'''
    1st and 2nd layers are single sublayers, the 2nd adds tuple pair nesting. Both are unpacked by func_pairs, not func_layers.  
    Multiple sublayers start on the 3rd layer, because it's derived from comparison between two (not one) lower layers. 
    4th layer is derived from comparison between 3 lower layers, where the 3rd layer is already nested, etc:
    initial 3-layer nesting: https://github.com/assets/52521979/ea6d436a-6c5e-429f-a152-ec89e715ebd6
    '''


# the below is out of date:

def sum_levels(Params, params):  # Capitalized names for sums, as comp_levels but no separate der_layers to return

    if Params:
        sum_players(Params[0], params[0])  # recursive unpack of nested ptuple layers, if any from der+
    else:
        Params.append(deepcopy(params[0]))  # no need to sum

    for Level, level in zip_longest(Params[1:], params[1:], fillvalue=[]):
        if Level and level:
            sum_levels(Level, level)  # recursive unpack of higher levels, if any from agg+ and nested with sub_levels
        elif level:
            Params.append(deepcopy(level))  # no need to sum


def form_PPP_t(iPPP_t):  # form PPs from match-connected segs
    PPP_t = []

    for fPd, iPPP_ in enumerate(iPPP_t):
        # sort by value of last layer: derivatives of all lower layers:
        if fPd:
            iPPP_ = sorted(iPPP_, key=lambda iPPP: iPPP.dval, reverse=True)  # descending order
        else:
            iPPP_ = sorted(iPPP_, key=lambda iPPP: iPPP.mval, reverse=True)  # descending order

        PPP_ = []
        for i, iPPP in enumerate(iPPP_):
            if fPd:
                iPPP_val = iPPP.dval
            else:
                iPPP_val = iPPP.mval

            for mptuple, dptuple in zip(iPPP.mplayer, iPPP.dplayer):
                if mptuple and dptuple:  # could be None
                    if fPd:
                        iPPP.rdn += dptuple.val > mptuple.val
                    else:
                        iPPP.rdn += mptuple.val > dptuple.val
            '''
            for param_layer in iPPP.params:  # may need recursive unpack here
                iPPP.rdn += sum_named_param(param_layer, 'val', fPd=fPd)> sum_named_param(param_layer, 'val', fPd=1-fPd)
            '''

            ave = vaves[fPd] * iPPP.rdn * (i + 1)  # derPP is redundant to higher-value previous derPPs in derPP_
            if iPPP_val > ave:
                PPP_ += [iPPP]  # base derPP and PPP is CPP
                if iPPP_val > ave * 10:
                    comp_PP_(iPPP, fPd)  # replace with reclustering?
            else:
                break  # ignore below-ave PPs
        PPP_t.append(PPP_)
    return PPP_t


def form_segPPP_root(PP_, root_rdn, fPd):  # not sure about form_seg_root

    for PP in PP_:
        link_eval(PP.uplink_layers, fPd)
        link_eval(PP.downlink_layers, fPd)

    for PP in PP_:
        form_segPPP_(PP)


def form_segPPP_(PP):
    pass


# pending update
def splice_segs(seg_):  # in 1st run of agg_recursion
    pass


# draft, splice 2 PPs for now
def splice_PPs(PP_, frng):  # splice select PP pairs if der+ or triplets if rng+

    spliced_PP_ = []
    while PP_:
        _PP = PP_.pop(0)  # pop PP, so that we can differentiate between tested and untested PPs
        tested_segs = []  # new segs may be added during splicing, their links also need to be checked for splicing
        _segs = _PP.seg_levels[0]

        while _segs:
            _seg = _segs.pop(0)
            _avg_y = sum([P.y for P in _seg.P__]) / len(_seg.P__)  # y centroid for _seg

            for link in _seg.uplink_layers[1] + _seg.downlink_layers[1]:
                seg = link.P.root  # missing link of current seg

                if seg.root is not _PP:  # after merging multiple links may have the same PP
                    avg_y = sum([P.y for P in seg.P__]) / len(seg.P__)  # y centroid for seg

                    # test for y distance (temporary)
                    if (_avg_y - avg_y) < ave_splice:
                        if seg.root in PP_:
                            PP_.remove(seg.root)  # remove merged PP
                        elif seg.root in spliced_PP_:
                            spliced_PP_.remove(seg.root)
                        # splice _seg's PP with seg's PP
                        merge_PP(_PP, seg.root)

            tested_segs += [_seg]  # pack tested _seg
        _PP.seg_levels[0] = tested_segs
        spliced_PP_ += [_PP]

    return spliced_PP_


def merge_PP(_PP, PP, fPd):  # only for PP splicing

    for seg in PP.seg_levels[fPd][-1]:  # merge PP_segs into _PP:
        accum_PP(_PP, seg, fPd)
        _PP.seg_levels[fPd][-1] += [seg]

    # merge uplinks and downlinks
    for uplink in PP.uplink_layers:
        if uplink not in _PP.uplink_layers:
            _PP.uplink_layers += [uplink]
    for downlink in PP.downlink_layers:
        if downlink not in _PP.downlink_layers:
            _PP.downlink_layers += [downlink]