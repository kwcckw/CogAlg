import cv2
from itertools import zip_longest
from copy import copy, deepcopy
import numpy as np
from frame_blobs import CBlob, flood_fill, assign_adjacents
from agg_recursion import *
from comp_slice import *

'''
comp_slice_ sub_recursion + utilities
'''
ave_rotate = 10
# n and val should be excluded?
PP_vars = ["I", "M", "Ma", "axis", "angle", "aangle", "G", "Ga", "x", "L"]

def sub_recursion_eval(root):  # for PP or dir_blob

    root_PPm_, root_PPd_ = root.rlayers[-1], root.dlayers[-1]
    for fd, PP_ in enumerate([root_PPm_, root_PPd_]):
        mcomb_layers, dcomb_layers, PPm_, PPd_ = [], [], [], []

        for PP in PP_:
            '''
            fd = _P.valt[1]+P.valt[1] > _P.valt[0]+_P.valt[0]  # if exclusive comp fork per latuple in P| vertuple in derP?
            '''
            if fd:  # add root to derP for der+:
                for P_ in PP.P__[1:-1]:  # skip 1st and last row
                    for P in P_:
                        for derP in P.uplink_layers[-1][fd]:
                            derP.roott[fd] = PP
                comb_layers = dcomb_layers; PP_layers = PP.dlayers; PPd_ += [PP]
            else:
                comb_layers = mcomb_layers; PP_layers = PP.rlayers; PPm_ += [PP]

            val = PP.valt[fd]; alt_val = sum([alt_PP.valt[fd] for alt_PP in PP.alt_PP_]) if PP.alt_PP_ else 0   # for fork rdn:
            ave = PP_aves[fd] * (PP.rdnt[fd] + 1 + (alt_val > val))
            if val > ave and len(PP.P__) > ave_nsub:
                sub_recursion(PP)  # comp_P_der | comp_P_rng in PPs -> param_layer, sub_PPs
                ave*=2  # 1+PP.rdn incr
                # splice deeper layers between PPs into comb_layers:
                for i, (comb_layer, PP_layer) in enumerate(zip_longest(comb_layers, PP_layers, fillvalue=[])):
                    if PP_layer:
                        if i > len(comb_layers) - 1: comb_layers += [PP_layer]  # add new r|d layer
                        else: comb_layers[i] += PP_layer  # splice r|d PP layer into existing layer

            # segs:
            agg_recursion_eval(PP, [copy(PP.mseg_levels[-1]), copy(PP.dseg_levels[-1])])

            # include empty comb_layers:
            if fd:
                PPmm_ = [PPm_] + mcomb_layers; mVal = sum([PP.valt[0] for PP_ in PPmm_ for PP in PP_])
                PPmd_ = [PPm_] + dcomb_layers; dVal = sum([PP.valt[1] for PP_ in PPmd_ for PP in PP_])
                root.dlayers = [PPmd_,PPmm_]
            else:
                PPdm_ = [PPm_] + mcomb_layers; mVal = sum([PP.valt[0] for PP_ in PPdm_ for PP in PP_])
                PPdd_ = [PPd_] + dcomb_layers; dVal = sum([PP.valt[1] for PP_ in PPdd_ for PP in PP_])
                root.rlayers = [PPdm_, PPdd_]
            # or higher der val?
            if isinstance(root, CPP):  # root is CPP
                for i in 0,1:
                    root.valt[i] += PP.valt[i]  # vals
                    root.rdnt[i] += PP.rdnt[i]  # ad rdn too?
            else:  # root is CBlob
                if fd: root.G += sum([alt_PP.valt[fd] for alt_PP in PP.alt_PP_]) if PP.alt_PP_ else 0
                else:  root.M += PP.valt[fd]


def sub_recursion(PP):  # evaluate each PP for rng+ and der+

    P__  = [P_ for P_ in reversed(PP.P__)]  # revert to top down
    P__ = comp_P_der(P__) if PP.fds[-1] else comp_P_rng(P__, PP.rng + 1)   # returns top-down
    PP.rdnt[PP.fds[-1] ] += 1  # two-fork rdn, priority is not known?  rotate?

    sub_segm_ = form_seg_root([copy(P_) for P_ in P__], fd=0, fds=PP.fds)
    sub_segd_ = form_seg_root([copy(P_) for P_ in P__], fd=1, fds=PP.fds)  # returns bottom-up
    # sub_PPm_, sub_PPd_:
    sub_PPm_, sub_PPd_ = form_PP_root((sub_segm_, sub_segd_), PP.rdnt[PP.fds[-1]] + 1)
    PP.rlayers[:] = [sub_PPm_]; PP.dlayers[:] = [sub_PPd_]

    sub_recursion_eval(PP)  # add rlayers, dlayers, seg_levels to select sub_PPs

def comp_P_rng(P__, rng):  # rng+ sub_recursion in PP.P__, switch to rng+n to skip clustering?

    for P_ in P__:
        for P in P_:  # add 2 link layers: rng_derP_ and match_rng_derP_:
            P.uplink_layers += [[],[[],[]]]; P.downlink_layers += [[],[[],[]]]

    for y, _P_ in enumerate(P__[:-rng]):  # higher compared row, skip last rng: no lower comparand rows
        for x, _P in enumerate(_P_):
            # get linked Ps at dy = rng-1:
            for pri_derP in _P.downlink_layers[-3][0]:  # fd always = 0 here
                pri_P = pri_derP.P
                # compare linked Ps at dy = rng:
                for ini_derP in pri_P.downlink_layers[0]:
                    P = ini_derP.P
                    # add new Ps, their link layers and reset their roots:
                    if P not in [P for P_ in P__ for P in P_]:
                        append_P(P__, P)
                        P.uplink_layers += [[],[[],[]]]; P.downlink_layers += [[],[[],[]]]
                    derP = comp_P(_P, P)
                    P.uplink_layers[-2] += [derP]
                    _P.downlink_layers[-2] += [derP]

    return P__

def comp_P_der(P__):  # der+ sub_recursion in PP.P__, compare P.uplinks to P.downlinks

    dderPs__ = []  # derP__ = [[] for P_ in P__[:-1]]  # init derP rows, exclude bottom P row

    for P_ in P__[1:-1]:  # higher compared row, exclude 1st: no +ve uplinks, and last: no +ve downlinks
        # not revised:
        dderPs_ = []  # row of dderPs
        for P in P_:
            dderPs = []  # dderP for each _derP, derP pair in P links
            for _derP in P.uplink_layers[-1][1]:  # fd=1
                for derP in P.downlink_layers[-1][1]:
                    # there maybe no x overlap between recomputed Ls of _derP and derP, compare anyway,
                    # mderP * (ave_olp_L / olp_L)? or olp(_derP._P.L, derP.P.L)?
                    # gap: neg_olp, ave = olp-neg_olp?
                    dderP = comp_P(_derP, derP)  # form higher vertical derivatives of derP.players,
                    # or comp derP.players[1] only: it's already diffs of all lower players?
                    derP.uplink_layers[0] += [dderP]  # pre-init layer per derP
                    _derP.downlink_layers[0] += [dderP]
                    dderPs_ += [dderP]  # actually it could be dderPs_ ++ [derPP]
                # compute x overlap between dderP'__P and P, in form_seg_ or comp_layer?
            dderPs_ += dderPs  # row of dderPs
        dderPs__ += [dderPs_]

    return dderPs__

def comp_P(_P, P):  # forms vertical derivatives of params per P in _P.uplink, conditional ders from norm and DIV comp

    if isinstance(_P, CP):
        vertuple = comp_ptuple(_P.ptuple, P.ptuple)
        derQ = [vertuple]; valt=copy(vertuple.valt); rdnt=copy(vertuple.rdnt)
        L = len(_P.dert_)
    else:  # P is derP
        derQ, rdnt, valt = comp_derH(_P.derQ, P.derQ)
        L = _P.L

    # derP is single-layer, links are compared individually?
    return CderP(derQ=derQ, valt=valt, rdnt=rdnt, P=P, _P=_P, x0=_P.x0, y0=_P.y0, L=L)

def rotate_P_(P__, dert__, mask__):  # rotate each P to align it with direction of P gradient

    yn, xn = dert__[0].shape[:2]
    for P_ in P__:
        for P in P_:
            daxis = P.ptuple.angle[0] / P.ptuple.L  # dy: deviation from horizontal axis
            while P.ptuple.G * abs(daxis) > ave_rotate:
                P.ptuple.axis = P.ptuple.angle
                rotate_P(P, dert__, mask__, yn, xn)  # recursive reform P along new axis in blob.dert__
                _, daxis = comp_angle("axis", P.ptuple.axis, P.ptuple.angle)
            # store P.daxis to adjust params?

def rotate_P(P, dert__t, mask__, yn, xn):

    L = len(P.dert_)
    rdert_ = [P.dert_[int(L/2)]]  # init rotated dert_ with old central dert

    ycenter = int(P.y0 + P.ptuple.axis[0]/2)  # can be negative
    xcenter = int(P.x0 + abs(P.ptuple.axis[1]/2))  # always positive
    Dy, Dx = P.ptuple.angle
    dy = Dy/L; dx = abs(Dx/L)  # hypot(dy,dx)=1: each dx,dy adds one rotated dert|pixel to rdert_
    # scan left:
    rx=xcenter-dx; ry=ycenter-dy; rdert=1  # to start while:
    while rdert and rx>=0 and ry>=0 and np.ceil(ry)<yn:
        rdert = form_rdert(rx,ry, dert__t, mask__)
        if rdert:
            rdert_.insert(0, rdert)
        rx += dx; ry += dy  # next rx, ry
    P.x0 = rx+dx; P.y0 = ry+dy  # revert to leftmost
    # scan right:
    rx=xcenter+dx; ry=ycenter+dy; rdert=1  # to start while:
    while rdert and ry>=0 and np.ceil(rx)<xn and np.ceil(ry)<yn:
        rdert = form_rdert(rx,ry, dert__t, mask__)
        if rdert:
            rdert_ += [rdert]
            rx += dx; ry += dy  # next rx,ry
    # form rP:
    # initialization:
    rdert = rdert_[0]; _, G, Ga, I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1 = rdert; M=ave_g-G; Ma=ave_ga-Ga; ndert_=[rdert]
    # accumulation:
    for rdert in rdert_[1:]:
        _, g, ga, i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = rdert
        I+=i; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx; Sin_da0+=sin_da0; Cos_da0+=cos_da0; Sin_da1+=sin_da1; Cos_da1+=cos_da1
        ndert_ += [rdert]
    # re-form gradients:
    G = np.hypot(Dy,Dx);  Ga = (Cos_da0 + 1) + (Cos_da1 + 1)
    ptuple = Cptuple(I=I, M=M, G=G, Ma=Ma, Ga=Ga, angle=(Dy,Dx), aangle=(Sin_da0, Cos_da0, Sin_da1, Cos_da1))
    # add n,val,L,x,axis?
    # replace P:
    P.ptuple=ptuple; P.dert_=ndert_


def form_rdert(rx,ry, dert__t, mask__):

    # coord, distance of four int-coord derts, overlaid by float-coord rdert in dert__, int for indexing
    # always in dert__ for intermediate float rx,ry:
    x1 = int(np.floor(rx)); dx1 = abs(rx - x1)
    x2 = int(np.ceil(rx));  dx2 = abs(rx - x2)
    y1 = int(np.floor(ry)); dy1 = abs(ry - y1)
    y2 = int(np.ceil(ry));  dy2 = abs(ry - y2)

    # scale all dert params in proportion to inverted distance from rdert, sum(distances) = 1?
    # approximation, square of rpixel is rotated, won't fully match not-rotated derts
    mask = mask__[y1, x1] * (1 - np.hypot(dx1, dy1)) \
         + mask__[y2, x1] * (1 - np.hypot(dx1, dy2)) \
         + mask__[y1, x2] * (1 - np.hypot(dx2, dy1)) \
         + mask__[y2, x2] * (1 - np.hypot(dx2, dy2))
    mask = int(mask)  # summed mask is fractional, round to 1|0
    if not mask:
        ptuple = []
        for dert__ in dert__t:  # 10 params in dert: i, g, ga, ri, dy, dx, day0, dax0, day1, dax1
            param = dert__[y1, x1] * (1 - np.hypot(dx1, dy1)) \
                  + dert__[y2, x1] * (1 - np.hypot(dx1, dy2)) \
                  + dert__[y1, x2] * (1 - np.hypot(dx2, dy1)) \
                  + dert__[y2, x2] * (1 - np.hypot(dx2, dy2))
            ptuple += [param]
        return ptuple


def append_P(P__, P):  # pack P into P__ in top down sequence

    current_ys = [P_[0].y0 for P_ in P__]  # list of current-layer seg rows
    if P.y0 in current_ys:
        P__[current_ys.index(P.y0)].append(P)  # append P row
    elif P.y0 > current_ys[0]:  # P.y0 > largest y in ys
        P__.insert(0, [P])
    elif P.y0 < current_ys[-1]:  # P.y0 < smallest y in ys
        P__.append([P])
    elif P.y0 < current_ys[0] and P.y0 > current_ys[-1]:  # P.y0 in between largest and smallest value
        for i, y in enumerate(current_ys):  # insert y if > next y
            if P.y0 > y: P__.insert(i, [P])  # PP.P__.insert(P.y0 - current_ys[-1], [P])


def copy_P(P, Ptype=None):  # Ptype =0: P is CP | =1: P is CderP | =2: P is CPP | =3: P is CderPP | =4: P is CaggPP

    if not Ptype:  # assign Ptype based on instance type if no input type is provided
        if isinstance(P, CPP):     Ptype = 2
        elif isinstance(P, CderP): Ptype = 1
        else:                      Ptype = 0  # CP

    uplink_layers, downlink_layers = P.uplink_layers, P.downlink_layers  # local copy of link layers
    P.uplink_layers, P.downlink_layers = [], []  # reset link layers
    roott = P.roott  # local copy
    P.roott = [None, None]
    if Ptype == 1:
        P_derP, _P_derP = P.P, P._P  # local copy of derP.P and derP._P
        P.P, P._P = None, None  # reset
    elif Ptype == 2:
        mseg_levels, dseg_levels = P.mseg_levels, P.dseg_levels
        rlayers, dlayers = P.rlayers, P.dlayers
        P__ = P.P__
        P.mseg_levels, P.dseg_levels, P.rlayers, P.dlayers, P.P__ = [], [], [], [], []  # reset
    elif Ptype == 3:
        PP_derP, _PP_derP = P.PP, P._PP  # local copy of derP.P and derP._P
        P.PP, P._PP = None, None  # reset
    elif Ptype == 4:
        gPP_, cPP_ = P.gPP_, P.cPP_
        rlayers, dlayers = P.rlayers, P.dlayers
        mlevels, dlevels = P.mlevels, P.dlevels
        P.gPP_, P.cPP_, P.rlayers, P.dlayers, P.mlevels, P.dlevels = [], [], [], [], [], []  # reset

    new_P = deepcopy(P)  # copy P with empty root and link layers, reassign link layers:
    new_P.uplink_layers += copy(uplink_layers)
    new_P.downlink_layers += copy(downlink_layers)

    # shallow copy to create new list
    P.uplink_layers, P.downlink_layers = copy(uplink_layers), copy(downlink_layers)  # reassign link layers
    P.roott = roott  # reassign root
    # reassign other list params
    if Ptype == 1:
        new_P.P, new_P._P = P_derP, _P_derP
        P.P, P._P = P_derP, _P_derP
    elif Ptype == 2:
        P.mseg_levels, P.dseg_levels = mseg_levels, dseg_levels
        P.rlayers, P.dlayers = rlayers, dlayers
        new_P.rlayers, new_P.dlayers = copy(rlayers), copy(dlayers)
        new_P.P__ = copy(P__)
        new_P.mseg_levels, new_P.dseg_levels = copy(mseg_levels), copy(dseg_levels)
    elif Ptype == 3:
        new_P.PP, new_P._PP = PP_derP, _PP_derP
        P.PP, P._PP = PP_derP, _PP_derP
    elif Ptype == 4:
        P.gPP_, P.cPP_ = gPP_, cPP_
        P.rlayers, P.dlayers = rlayers, dlayers
        P.roott = roott
        new_P.gPP_, new_P.cPP_ = [], []
        new_P.rlayers, new_P.dlayers = copy(rlayers), copy(dlayers)
        new_P.mlevels, new_P.dlevels = copy(mlevels), copy(dlevels)

    return new_P

def frame2graph(frame, fseg, Cgraph):  # for frame_recursive

    mblob_ = frame.PPm_; dblob_ = frame.PPd_  # PPs are blobs here
    x0, xn, y0, yn = frame.box
    gframe = Cgraph(alt_plevels=CpH, rng=mblob_[0].rng, rdn=frame.rdn, x0=(x0+xn)/2, xn=(xn-x0)/2, y0=(y0+yn)/2, yn=(yn-y0)/2)
    for fd, blob_, plevels in enumerate(zip([mblob_,dblob_], [gframe.plevels, gframe.alt_plevels])):
        graph_ = []
        for blob in blob_:
            graph = PP2graph(blob, fseg, Cgraph, fd)
            sum_pH(plevels, graph.plevels)
            graph_ += [graph]
        [gframe.node_.Q, gframe.alt_graph_][fd][:] = graph_  # mblob_|dblob_, [:] to enable to left hand assignment, not valid for object

    return gframe

# tentative, will be finalized when structure in agg+ is finalized
def blob2graph(blob, fseg):

    PPm_ = blob.PPm_; PPd_ = blob.PPd_
    x0, xn, y0, yn = blob.box

    alt_mblob = Cgraph(fds=copy(PPm_[0].fds), aggH=CQ(Q=[CQ(Q=[])]), rng=PPm_[0].rng, box=[(y0+yn)/2,(x0+xn)/2, y0,yn, x0,xn])
    alt_dblob = Cgraph(fds=copy(PPd_[0].fds), aggH=CQ(Q=[CQ(Q=[])]), rng=PPm_[0].rng, box=[(y0+yn)/2,(x0+xn)/2, y0,yn, x0,xn])

    mblob = Cgraph(fds=copy(PPm_[0].fds), aggH=CQ(Q=[CQ(Q=[])]), alt_Graph=alt_mblob, rng=PPm_[0].rng, box=[(y0+yn)/2,(x0+xn)/2, y0,yn, x0,xn])
    dblob = Cgraph(fds=copy(PPd_[0].fds), aggH=CQ(Q=[CQ(Q=[])]), alt_Graph=alt_dblob, rng=PPd_[0].rng, box=[(y0+yn)/2,(x0+xn)/2, y0,yn, x0,xn])

    blob.mgraph = mblob  # update graph reference
    blob.dgraph = dblob  # update graph reference
    blobs= [mblob, dblob]

    for fd, PP_ in enumerate([PPm_,PPd_]):  # if any
        for PP in PP_:
            graph = PP2graph(PP, fseg, fd)
            sum_aggH(blobs[fd].aggH, graph.aggH)
            for i in range(2):
                blobs[fd].valt[i] += graph.valt[i]
                blobs[fd].rdnt[i] += graph.rdnt[i]
            graph.root = blobs[fd]
            blobs[fd].node_ += [graph]

    for alt_blob in blob.adj_blobs[0]:  # adj_blobs = [blobs, pose]
        if not alt_blob.mgraph:
            blob2graph(alt_blob, fseg)  # convert alt_blob to graph
        sum_aggH(alt_mblob.aggH, alt_blob.mgraph.aggH)
        sum_aggH(alt_dblob.aggH, alt_blob.dgraph.aggH)
        for i in range(2):
            alt_mblob.valt[i] += alt_blob.mgraph.valt[i]
            alt_mblob.rdnt[i] += alt_blob.mgraph.rdnt[i]
            alt_dblob.valt[i] += alt_blob.dgraph.valt[i]
            alt_dblob.rdnt[i] += alt_blob.dgraph.rdnt[i]

    return mblob, dblob


# tentative, will be finalized when structure in agg+ is finalized
def PP2graph(PP, fseg, ifd=1):

    alt_derH = CQ(fds=0); alt_subH = CQ(Q=[alt_derH],fds=[0]); alt_aggH = CQ(Q=[alt_subH],fds=[0]); alt_valt = [0,0]; alt_rdnt = [0,0]; alt_box = [0,0,0,0]
    if not fseg and PP.alt_PP_:  # seg doesn't have alt_PP_
        alt_derH.Q = [deepcopy(PP.alt_PP_[0].derH)]; alt_valt = copy(PP.alt_PP_[0].valt)
        alt_box = copy(PP.alt_PP_[0].box); alt_rdnt = copy(PP.alt_PP_[0].rdnt)
        for altPP in PP.alt_PP_[1:]:  # get fd sequence common for all altPPs:
            sum_derH(alt_derH.Q[0], altPP.derH)
            Y0,Yn,X0,Xn = alt_box; y0,yn,x0,xn = altPP.box
            alt_box[:] = min(Y0,y0),max(Yn,yn),min(X0,x0),max(Xn,xn)
            for i in range(2):
                alt_valt[i] += altPP.valt[i]
                alt_rdnt[i] += altPP.rdnt[i]
    alt_Graph = Cgraph(aggH=alt_aggH, valt=alt_valt, rdnt=alt_rdnt, box=alt_box)

    derH = CQ(Q=PP.derH, fds=[0])
    subH = CQ(Q=[derH],fds=[0]); aggH = CQ(Q=[subH],fds=[0])
    graph = Cgraph(aggH=aggH, valt=copy(PP.valt), rndt=copy(PP.rdnt), box=copy(PP.box), alt_Graph=alt_Graph)

    return graph

# drafts:
def inpack_derH(pPP, ptuples, idx_=[]):  # pack ptuple vars in derH of pPP vars, converting macro derH -> micro derH
    # idx_: indices per lev order in derH, lenlev: 1, 1, 2, 4, 8...

    repack(pPP, ptuples[0], idx_+[0])  # single-element 1st lev
    if len(ptuples)>1:
        repack(pPP, ptuples[1], idx_+[1])  # single-element 2nd lev
        i=2; last=4
        idx = 2  # init incremental elevation = i
        while last<=len(ptuples):
            lev = ptuples[i:last]  # lev is nested, max len_sublev = lenlev-1, etc.
            inpack_derH(pPP, lev, idx_+[idx])  # add idx per sublev
            i=last; last+=i  # last=i*2
            idx+=1  # elevation in derH

def repack(pPP, ptuple, idx_):  # pack derH in elements of iderH

    for i, param_name in enumerate(PP_vars):
        par = getattr(ptuple, param_name)
        Par = pPP[i]
        if len(Par) > len(idx_):  # Par is derH of pars
            Par[-1] += [par]  # pack par in top lev of Par, added per inpack_derH recursion
        else:
            Par += [[par]]  # add new Par lev, implicitly nested in ptuples?

# move here temporary, for debug purpose
def agg_recursion_eval(blob, PP_t):
    from agg_recursion import agg_recursion
    from sub_recursion import PP2graph, blob2graph

    fseg = isinstance(blob, CPP)

    for fd, PP_ in enumerate(PP_t):
        for i, PP in enumerate(PP_):
           converted_graph  = PP2graph(PP, fseg=fseg, ifd=fd)  # convert PP to graph
           PP_[i] = converted_graph
    if fseg:
        converted_mblob = PP2graph(blob, fseg=fseg, ifd=0)  # convert root to graph (root default fd = 1?)
        converted_dblob = PP2graph(blob, fseg=fseg, ifd=1)  # when fseg = True, we need both forks?
        converted_mblob.node_ = PP_t[0]; converted_dblob.node_ = PP_t[1]
        converted_blobt = [converted_mblob,converted_dblob]
        for PP in PP_t[0]: PP.root = converted_blobt[0]
        for PP in PP_t[1]: PP.root = converted_blobt[1]
    else:
        if blob.mgraph:
            converted_blobt = [blob.mgraph, blob.dgraph]  # get converted graph
        else:
            converted_blobt = blob2graph(blob, fseg=fseg)  # convert root to graph

    M = sum(converted_blobt[0].valt)  # mpplayers.val (but m fork is always empty, so no value here?)
    G = sum(converted_blobt[1].valt)  # dpplayers.val
    valt = [M, G]
    fork_rdnt = [1+(G>M), 1+(M>=G)]
    # should be single call of agg_recursion here？
    for fd, PP_ in enumerate(PP_t):  # PPm_, PPd_
        if (valt[fd] > PP_aves[fd] * ave_agg * (converted_blobt[fd].rdnt[fd]+1) * fork_rdnt[fd]) \
            and len(PP_) > ave_nsub : # and converted_blob[0].alt_rdn < ave_overlap:

            blob.rdn += 1  # estimate
            agg_recursion(converted_blobt[fd], fseg=fseg)