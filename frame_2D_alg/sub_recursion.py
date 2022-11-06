from itertools import zip_longest
from comp_slice import CPP,CP, CderP
from comp_slice import form_seg_root,form_PP_root, append_P, comp_P
from comp_slice import ave_nsub, ave_splice, PP_aves
from agg_recursion import agg_recursion_eval
from copy import copy, deepcopy


def sub_recursion_eval(root):  # for PP or dir_blob

    if isinstance(root, CPP): root_PPm_, root_PPd_ = root.rlayers[0], root.dlayers[0]
    else:                     root_PPm_, root_PPd_ = root.PPm_, root.PPd_

    for fd, PP_ in enumerate([root_PPm_, root_PPd_]):
        mcomb_layers, dcomb_layers, PPm_, PPd_ = [], [], [], []

        for PP in PP_:
            if fd:  # add root to derP for der+:
                for P_ in PP.P__[1:-1]:  # skip 1st and last row
                    for P in P_:
                        for derP in P.uplink_layers[-1][fd]:
                            derP.roott[fd] = PP
                comb_layers = dcomb_layers; PP_layers = PP.dlayers; PPd_ += [PP]
            else:
                comb_layers = mcomb_layers; PP_layers = PP.rlayers; PPm_ += [PP]

            val = PP.valt[fd]; alt_val = PP.valt[1-fd]  # for fork rdn:
            ave = PP_aves[fd] * (PP.rdn + 1 + (alt_val > val))
            if val > ave and len(PP.P__) > ave_nsub:
                sub_recursion(PP)  # comp_P_der | comp_P_rng in PPs -> param_layer, sub_PPs
                ave*=2  # 1+PP.rdn incr
                # splice deeper layers between PPs into comb_layers:
                for i, (comb_layer, PP_layer) in enumerate(zip_longest(comb_layers, PP_layers, fillvalue=[])):
                    if PP_layer:
                        if i > len(comb_layers) - 1: comb_layers += [PP_layer]  # add new r|d layer
                        else: comb_layers[i] += PP_layer  # splice r|d PP layer into existing layer

            # segs agg_recursion:
            agg_recursion_eval(PP, [copy(PP.mseg_levels[-1]), copy(PP.dseg_levels[-1])])
            # include empty comb_layers:
            if fd: root.dlayers = [[[PPm_] + mcomb_layers], [[PPd_] + dcomb_layers]]
            else:  root.rlayers = [[[PPm_] + mcomb_layers], [[PPd_] + dcomb_layers]]

            # or higher der val?
            if isinstance(root, CPP):  # root is CPP
                root.valt[fd] += PP.valt[fd]
            else:  # root is CBlob
                if fd: root.G += PP.valt[1]
                else:  root.M += PP.valt[0]


def sub_recursion(PP):  # evaluate each PP for rng+ and der+

    P__  = [P_ for P_ in reversed(PP.P__)]  # revert to top down
    P__ = comp_P_der(P__) if PP.fds[-1] else comp_P_rng(P__, PP.rng + 1)   # returns top-down
    PP.rdn += 2  # two-fork rdn, priority is not known?

    sub_segm_ = form_seg_root([copy(P_) for P_ in P__], fd=0, fds=PP.fds)
    sub_segd_ = form_seg_root([copy(P_) for P_ in P__], fd=1, fds=PP.fds)  # returns bottom-up
    # sub_PPm_, sub_PPd_:
    PP.rlayers[0], PP.dlayers[0] = form_PP_root((sub_segm_, sub_segd_), PP.rdn + 1)
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

# draft
def splice_dir_blob_(dir_blobs):
    for i, _dir_blob in enumerate(dir_blobs):  # it may be redundant to loop all blobs here, use pop should be better here
        for fd in 0, 1:
            if fd: PP_ = _dir_blob.PPd_
            else:  PP_ = _dir_blob.PPm_
            PP_val = sum([PP.valt[fd] for PP in PP_])

            if PP_val - ave_splice > 0:  # high mPP pr dPP
                _top_P_ = _dir_blob.P__[0]
                _bottom_P_ = _dir_blob.P__[-1]
                for j, dir_blob in enumerate(dir_blobs):
                    if _dir_blob is not dir_blob:

                        top_P_ = dir_blob.P__[0]
                        bottom_P_ = dir_blob.P__[-1]
                        # test y adjacency:
                        if (_top_P_[0].y - 1 == bottom_P_[0].y) or (top_P_[0].y - 1 == _bottom_P_[0].y):
                            # test x overlap:
                            if (dir_blob.x0 - 1 < _dir_blob.xn and dir_blob.xn + 1 > _dir_blob.x0) \
                                    or (_dir_blob.x0 - 1 < dir_blob.xn and _dir_blob.xn + 1 > dir_blob.x0):
                                splice_2dir_blobs(_dir_blob, dir_blob)  # splice dir_blob into _dir_blob
                                dir_blobs[j] = _dir_blob

def splice_2dir_blobs(_blob, blob):
    # merge blob into _blob here
    
    # P__, box, dert and mask
    # not sure how to merge dert__ and mask__ yet
    x0 = min(_blob.box[0], blob.box[0]) 
    xn = max(_blob.box[1], blob.box[1]) 
    y0 = min(_blob.box[2], blob.box[2]) 
    yn = max(_blob.box[3], blob.box[3]) 
    _blob.box = (x0, xn, y0, yn)
    
    if (_blob.P__[0][0].y - 1 == blob.P__[-1][0].y):  # blob at top
        _blob.P__ = blob.P__ + _blob.P__
    else:  # _blob at top
        _blob.P__ = _blob.P__ + blob.P__ 

    # accumulate blob numeric params:
    # 'I', 'Dy', 'Dx', 'G', 'A', 'M', 'Sin_da0', 'Cos_da0', 'Sin_da1', 'Cos_da1', 'Ga', 'Mdx', 'Ddx', 'rdn', 'rng'
    for param_name in blob.numeric_params:
        _param = getattr(_blob,param_name)
        param = getattr(blob,param_name)
        setattr(_blob, param_name, _param+param)
       
    # accumulate blob list params:
    _blob.adj_blobs += blob.adj_blobs
    _blob.rlayers += blob.rlayers
    _blob.dlayers += blob.dlayers
    _blob.PPm_ += blob.PPm_
    _blob.PPd_ += blob.PPd_
    # _blob.valt[0] += blob.valt[0]; _blob.valt[1] += blob.valt[1] (we didnt assign valt yet)
    _blob.dir_blobs += blob.dir_blobs
    _blob.mlevels += blob.mlevels
    _blob.dlevels += blob.dlevels
     

def copy_P(P, iPtype=None):  # Ptype =0: P is CP | =1: P is CderP | =2: P is CPP | =3: P is CderPP | =4: P is CaggPP

    if not iPtype:  # assign Ptype based on instance type if no input type is provided
        if isinstance(P, CPP):
            Ptype = 2
        elif isinstance(P, CderP):
            Ptype = 1
        elif isinstance(P, CP):
            Ptype = 0
    else:
        Ptype = iPtype

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