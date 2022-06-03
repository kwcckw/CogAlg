'''
Comp_slice is a terminal fork of intra_blob.
-
It traces blob axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)
-
Vectorization is clustering of parameterized Ps + their derivatives (derPs) into PPs: patterns of Ps that describe edge blob.
This process is a reduced-dimensionality (2D->1D) version of cross-comp and clustering cycle, common across this project.
As we add higher dimensions (2D alg, 3D alg), this dimensionality reduction is done in salient high-aspect blobs
(likely edges / contours in 2D or surfaces in 3D) to form more compressed "skeletal" representations of full-D patterns.
'''

from collections import deque
import sys
import numpy as np
from itertools import zip_longest
from copy import deepcopy, copy
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
from segment_by_direction import segment_by_direction
# import warnings  # to detect overflow issue, in case of infinity loop
# warnings.filterwarnings('error')

ave_inv = 20  # ave inverse m, change to Ave from the root intra_blob?
ave = 5  # ave direct m, change to Ave_min from the root intra_blob?
ave_g = 30  # change to Ave from the root intra_blob?
ave_ga = 0.78  # ga at 22.5 degree
flip_ave = .1
flip_ave_FPP = 0  # flip large FPPs only (change to 0 for debug purpose)
div_ave = 200
ave_rmP = .7  # the rate of mP decay per relative dX (x shift) = 1: initial form of distance
ave_ortho = 20
aveB = 50
# comp_param coefs:
ave_I = ave_inv
ave_M = ave  # replace the rest with coefs:
ave_Ma = 10
ave_G = 10
ave_Ga = 2  # related to dx?
ave_L = 10
ave_dx = 5  # difference between median x coords of consecutive Ps
ave_dangle = 2  # vertical difference between angles
ave_daangle = 2
ave_mP = 10
ave_dP = 10
ave_mPP = 10
ave_dPP = 10
ave_splice = 10

param_names = ["x", "I", "M", "Ma", "L", "angle", "aangle"]  # angle = Dy, Dx; aangle = sin_da0, cos_da0, sin_da1, cos_da1; recompute Gs for comparison?
aves = [ave_dx, ave_I, ave_M, ave_Ma, ave_L, ave_G, ave_Ga, ave_mP, ave_dP]
vaves = [ave_mP, ave_dP]

class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP

    params = list  # 9 compared horizontal params: x, L, I, M, Ma, G, Ga, Ds( Dy, Dx, Sin_da0), Das( Cos_da0, Sin_da1, Cos_da1)
    # I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1 are summed from dert[3:], M, Ma from ave- g, ga
    # G, Ga are recomputed from Ds, Das; M, Ma are not restorable from G, Ga
    x0 = int
    x = float  # median x
    L = int
    sign = NoneType  # g-ave + ave-ga sign
    # all the above are redundant to params
    # rdn = int  # blob-level redundancy, ignore for now
    y = int  # for vertical gap in PP.P__
    # composite params:
    dert_ = list  # array of pixel-level derts, redundant to uplink_, only per blob?
    uplink_layers = lambda: [[],[]]  # init 1st 2 layers: derPs and match_derPs
    downlink_layers = lambda: [[],[]]
    root = object  # segment that contains this P
    # only in Pd:
    Pm = object  # reference to root P
    dxdert_ = list
    # only in Pm:
    Pd_ = list
    # if comp_dx:
    Mdx = int
    Ddx = int

class CderP(ClusterStructure):  # tuple of derivatives in P uplink_ or downlink_

    # dP, mP are packed in params[0,1]
    params = list  # P derivation layer, n_params = 9 * 2**der_cnt, flat, decoded by mapping each m,d to lower-layer param
    x0 = int  # redundant to params:
    x = float  # median x
    L = int  # pack in params?
    sign = NoneType  # g-ave + ave-ga sign
    y = int  # for vertical gaps in PP.P__, replace with derP.P.y?
    P = object  # lower comparand
    _P = object  # higher comparand
    root = object  # segment if internal or PP if external derP?
    # higher derivatives
    rdn = int  # mrdn, + uprdn if branch overlap?
    uplink_layers = lambda: [[],[]]  # init 1st 2 layers: dderPs and match_dderPs
    downlink_layers = lambda: [[],[]]
   # from comp_dx
    fdx = NoneType

class CPP(CP, CderP):  # P and derP params are combined into param_layers?

    params = list  # derivation layers += derP params per der+, param L is actually Area
    sign = bool
    rng = lambda: 1  # rng starts with 1
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs
    Rdn = int  # for accumulation only
    nP = int  # len 2D derP__ in levels[0][fPd]?  ly = len(derP__), also x, y?
    nderP = int
    uplink_layers = lambda: [[], []]
    downlink_layers = lambda: [[], []]
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    box = list  # for visualization only, original box before flipping
    mask__ = bool
    P__ = list  # input  # derP__ = list  # redundant to P__
    seg_levels = lambda: [[[]],[[]]]  # from 1st agg_recursion, seg_levels[0] is seg_t, higher seg_levels are segP_t s
    PPP_levels = list  # from 2nd agg_recursion, PP_t = levels[0], from form_PP, before recursion
    layers = list  # from sub_recursion, each is derP_t
    root = object  # higher-order segP or PPP

# Functions:

def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    segment_by_direction(blob, verbose=False)  # forms blob.dir_blobs
    for dir_blob in blob.dir_blobs:  # dir_blob should be CBlob

        P__ = slice_blob(dir_blob, verbose=False)  # cluster dir_blob.dert__ into 2D array of blob slices
        # comp_dx_blob(P__), comp_dx?

        Pm__ = comp_P_root(deepcopy(P__))  # scan_P_, comp_P | link_layer, adds mixed uplink_, downlink_ per P,
        Pd__ = comp_P_root(deepcopy(P__))  # deepcopy before assigning link derPs to Ps

        segm_ = form_seg_root(Pm__, root_rdn=2, fPd=0)  # forms segments: parameterized stacks of (P,derP)s
        segd_ = form_seg_root(Pd__, root_rdn=2, fPd=1)  # seg is a stack of (P,derP)s

        PPm_, PPd_ = form_PP_root((segm_, segd_), root_rdn=2)  # forms PPs: parameterized graphs of linked segs
        # temporary, pending update
        '''
        draw_seg_(dir_blob, [segm_, segd_])
        draw_PP_segs(dir_blob, PPm_)
        draw_PPs(dir_blob, PPm_, fspliced=0)
        '''
        
        sub_recursion([], PPd_, frng=0)  # der+ comp_P in PPds, -> param_layer, form sub_PPs
        sub_recursion([], PPm_, frng=1)  # rng+ comp_P in PPms, -> param_layer, form sub_PPs

        for PP_ in (PPm_, PPd_):  # 1st agglomerative recursion is per PP, appending PP.seg_levels, not blob.levels:
            for PP in PP_:
                agg_recursion(PP, fseg=1)  # higher-composition comp_seg -> segPs.. per seg__[n], in PP.seg_levels
        dir_blob.levels = [[PPm_], [PPd_]]
        agg_recursion(dir_blob, fseg=0)  # 2nd call per dir_blob.PP_s formed in 1st call, forms PPP..s and dir_blob.levels

    splice_dir_blob_(blob.dir_blobs)


def slice_blob(blob, verbose=False):  # forms horizontal blob slices: Ps, ~1D Ps, in select smooth edge (high G, low Ga) blobs

    mask__ = blob.mask__  # same as positive sign here
    dert__ = zip(*blob.dert__)  # convert 10-tuple of 2D arrays into 1D array of 10-tuple blob rows
    dert__ = [zip(*dert_) for dert_ in dert__]  # convert 1D array of 10-tuple rows into 2D array of 10-tuples per blob

    height, width = mask__.shape
    if verbose: print("Converting to image...")
    P__ = []  # blob of Ps

    for y, (dert_, mask_) in enumerate(zip(dert__, mask__)):  # unpack lines
        P_ = []  # line of Ps
        _mask = True
        for x, (dert, mask) in enumerate(zip(dert_, mask_)):  # unpack derts: tuples of 10 params
            if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

            if not mask:  # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination
                if _mask:  # initialize P params with first unmasked dert:
                    Pdert_ = [dert]
                    params = [ave_g-dert[1], ave_ga-dert[2], *dert[3:]]  # m, ma, dert[3:]: i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1
                else:
                    # dert and _dert are not masked, accumulate P params from dert params:
                    params[1] += ave_g-dert[1]; params[2] += ave_ga-dert[2]  # M, Ma
                    for i, (Param, param) in enumerate(zip(params[2:], dert[3:]), start=2):  # I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1
                        params[i] = Param + param
                    Pdert_.append(dert)
            elif not _mask:
                # _dert is not masked, dert is masked, terminate P:
                L = len(Pdert_)
                P_.append( CP(params= [x-(L-1)/2, L] + list(params), x0=x-(L-1), L=L, y=y, dert_=Pdert_))

            _mask = mask
        if not _mask:  # pack last P:
            L = len(Pdert_)
            P_.append( CP(params = [x-(L-1)/2, L] + list(params), x0=x-(L-1), L=L, y=y, dert_=Pdert_))
        P__ += [P_]

    blob.P__ = P__
    return P__


def comp_P_root(P__):  # vertically compares y-adjacent and x-overlapping Ps: blob slices, forming 2D derP__

    _P_ = P__[0]  # upper row, top-down
    for P_ in P__[1:]:  # lower row
        for P in P_:
            for _P in _P_:  # test for x overlap(_P,P) in 8 directions, derts are positive in all Ps:
                if (P.x0 - 1 < _P.x0 + _P.L) and (P.x0 + P.L + 1 > _P.x0):
                    derP = comp_P(_P, P)
                    P.uplink_layers[-2] += [derP]  # append derPs, uplink_layers[-1] is match_derPs
                    _P.downlink_layers[-2] += [derP]
                elif (P.x0 + P.L) < _P.x0:
                    break  # no P xn overlap, stop scanning lower P_
        _P_ = P_
    return P__


def comp_P_rng(iP__, rng):  # rng+ sub_recursion in PP.P__, adding two link_layers per P

    P__ = [P_ for P_ in reversed(iP__)]  # revert to top-down
    uplinks__ = [[ [] for P in P_] for P_ in P__[rng:]]  # rng derP_s per P, exclude 1st rng rows
    downlinks__ = [[ [] for P in P_] for P_ in P__[:-rng]]  # exclude last rng rows

    for y, _P_ in enumerate(P__[:-rng]):  # higher compared row, skip last rng: no lower comparand rows
        for x, _P in enumerate(_P_):
            for derP in _P.downlink_layers[-1]:  # lower comparands are linked Ps at dy = rng
                P = derP.P
                if isinstance(P, CPP) or isinstance(P, CderP):  # rng+ fork for derPs, very unlikely
                    derP = comp_derP(_P, P)  # form higher vertical derivatives of derP or PP params
                else:
                    derP = comp_P(_P, P)  # form vertical derivatives of horizontal P params
                # += links:
                downlinks__[y][x] += [derP]
                up_x = P__[y+rng].index(P)  # index of P in P_ at y+rng
                uplinks__[y][up_x] += [derP]  # uplinks__[y] = P__[y+rng]: uplinks__= P__[rng:]

    for P_, uplinks_ in zip( P__[rng:], uplinks__):  # skip 1st rmg rows, no uplinks
        for P, uplinks in zip(P_, uplinks_):
            P.uplink_layers += [uplinks, []]  # add two P.link_layers: rng_derP_ and empty match_rng_derP_

    for P_, downlinks_ in zip(P__[:-rng], downlinks__):  # skip last rng rows, no downlinks
        for P, downlinks in zip(P_, downlinks_):
            P.downlink_layers += [downlinks, []]

    return iP__  # return bottom-up P__


def comp_P_der(iP__):  # der+ sub_recursion in PP.P__, compare P.uplinks to P.downlinks

    P__ = [P_ for P_ in reversed(iP__)]  # revert to top-down
    dderPs__ = []  # derP__ = [[] for P_ in P__[:-1]]  # init derP rows, exclude bottom P row

    for P_ in P__[1:-1]:  # higher compared row, exclude 1st: no +ve uplinks, and last: no +ve downlinks
        dderPs_ = []  # row of dderPs
        for P in P_:
            dderPs = []  # dderP for each _derP, derP pair in P links
            for _derP in P.uplink_layers[-1]:
                for derP in P.downlink_layers[-1]:
                    # there maybe no x overlap between recomputed Ls of _derP and derP, compare anyway,
                    # mderP * (ave_olp_L / olp_L)? or olp(_derP._P.L, derP.P.L)?
                    # gap: neg_olp, ave = olp-neg_olp?
                    dderP = comp_derP(_derP, derP)  # form higher vertical derivatives of derP or PP params
                    derP.uplink_layers[0] += [dderP]  # pre-init layer per derP
                    _derP.downlink_layers[0] += [dderP]
                    dderPs += [dderP]
                # compute x overlap between dderP'__P and P, in form_seg_ or comp_layer?
            dderPs_ += dderPs  # row of dderPs
        dderPs__ += [dderPs_]

    return dderPs__


def form_seg_root(P__, root_rdn, fPd):  # form segs from Ps

    seg_ = []
    for P_ in reversed(P__):  # get a row of Ps bottom-up, different copies per fPd
        while P_:
            P = P_.pop(0)
            if P.uplink_layers[-2]:  # last derPs layer is not empty
                form_seg_(seg_, P__, [P], fPd)  # test P.matching_uplink_, not known in form_seg_root
            else:
                seg_.append( sum2seg([P], [], fPd))  # no link_s, terminate seg_Ps = [P]
    return seg_


def form_seg_(seg_, P__, seg_Ps, fPd):  # form contiguous segments of vertically matching derPs

    uplink_layer = sorted(seg_Ps[-1].uplink_layers[-2], key=lambda derP:derP.params[fPd], reverse=False)
    miss_uplink_ = []

    for i, derP in enumerate(uplink_layer):  # scan bottom-up to find match|miss links between compared Ps

        if fPd: derP.rdn += derP.params[0] > derP.params[1]  # mP > dP
        else:   rng_eval(derP, fPd)  # reset derP.val, derP.rdn
        if derP.params[fPd] > vaves[fPd] * (derP.rdn + len(derP.P.uplink_layers[-1])):  # val > ave * branch rdn,
            # the weaker links are redundant to the stronger, added to derP.P.uplink_layers[-1]) in prior loops
            derP.P.uplink_layers[-1].append(derP)
            derP._P.downlink_layers[-1].append(derP)
        else:
            miss_uplink_ = uplink_layer[i:]  # = seg.uplink_ if term, also add to higher-seg.downlink_ in sum2seg
            break  # the rest of uplinks is even weaker

    if len(seg_Ps[-1].uplink_layers[-1]) > 1:  # terminate seg
        seg_.append( sum2seg( seg_Ps, miss_uplink_, fPd)) # convert seg_Ps to CPP seg
    else:
        match_uplink_ = seg_Ps[-1].uplink_layers[-1]
        if match_uplink_ and len(match_uplink_[0]._P.downlink_layers[-1])==1:
            # one P.uplink AND one _P.downlink: add _P to seg, match_uplink_[0] is a sole uplinked derP:
            P = match_uplink_[0]._P
            [P_.remove(P) for P_ in P__ if P in P_]  # remove P from P__ so it's not inputted in form_seg_root
            seg_Ps += [P]  # if P.downlinks in seg_down_misses += [P]

            if seg_Ps[-1].uplink_layers[-2]:
                form_seg_(seg_, P__, seg_Ps, fPd)  # recursive compare sign of next-layer uplinks
            else:
                seg_.append( sum2seg(seg_Ps, miss_uplink_, fPd))
        else:
            seg_.append( sum2seg(seg_Ps, miss_uplink_, fPd))  # terminate seg at 0 matching uplink


def rng_eval(derP, fPd):  # compute value of combined mutual derPs: overlap between P uplinks and _P downlinks

    _P, P = derP._P, derP.P
    common_derP_ = []

    for _downlink_layer, uplink_layer in zip(_P.downlink_layers, P.uplink_layers):  # overlap in P uplinks and _P downlinks
        common_derP_ += list( set(_downlink_layer).intersection(uplink_layer))  # get common derP in mixed uplinks
    rdn = 1
    olp_val = 0
    for derP in common_derP_:
        rdn += derP.params[fPd] > derP.params[1-fPd]  # dP > mP if fPd, else mP > dP
        olp_val += derP.params[fPd]

    nolp = len(common_derP_)
    derP.params[fPd] = olp_val / nolp
    derP.rdn += (rdn / nolp) > .5  # no fractional rdn?


def sum2seg(seg_Ps, miss_uplink_, fPd):  # sum params of vertically connected Ps into segment

    downlinks = seg_Ps[0].downlink_layers[-2]  # downlinks of bottom P, downlink.P.seg.uplinks: lower seg.uplinks,

    # miss is computed for lower seg.uplinks to terminate and form new seg, but they overlap higher seg.downlinks:
    miss_downlink_ = [downlink for downlink in downlinks if downlink in downlink.P.root.uplink_layers[-1]]

    # also re-eval downward branches for redundancy?

    seg = CPP(x0=seg_Ps[0].x0, P__=seg_Ps, uplink_layers=[miss_uplink_], downlink_layers = [miss_downlink_],
              L = len(seg_Ps), y0 = seg_Ps[0].y)  # seg.L is Ly
    if isinstance(seg_Ps[0], CPP): accum_P = accum_CPP
    else: accum_P = accum_CP

    for P in seg_Ps[:-1]:
        accum_P(seg, P, fPd)  # sum P params into seg.params[:-1], layered in CPP
        accum_layer(seg.params[-1], P.uplink_layers[-1][0].params)  # sum single-derP params into top seg param layer
    accum_P(seg, seg_Ps[-1], fPd)  # accum top P, not top derP

    return seg

def sum2PP(PP_segs, fPd):  # sum params: derPs into segment or segs into PP

    PP = CPP(x0=PP_segs[0].x0, sign=PP_segs[0].sign,L= len(PP_segs))
    PP.seg_levels[fPd][0] = PP_segs  # PP_segs is seg_levels[0]

    for seg in PP_segs:
        accum_CPP(PP, seg, fPd)

    return PP

def accum_CP(seg, P, fPd):

    if not seg.params:
        seg.params = [P.params]  # single param layer
    else:
        accum_layer(seg.params[-1], P.params)  # P.params is top layer
    P.root = seg
    seg.x0 = min(seg.x0, P.x0)

def accum_CPP(PP, inp, fPd):  # inp is seg or PP in recursion

    if not PP.params:
        PP.params = inp.params.copy()
    else:
        for i, layer in enumerate(inp.params):  # params are param layers
            accum_layer(PP.params[i], layer)

    PP.x0 = min(PP.x0, inp.x0)
    PP.Rdn += inp.rdn  # root_rdn + PP.Rdn / PP: recursion rdn + average (forks + links) rdn
    # link rdn: nderP / len(P__)?
    PP.y = max(inp.y, PP.y)  # or pass local y arg instead of derP.y?
    inp.root = PP
    PP.nderP += len(inp.P__[-1].uplink_layers[-1])  # redundant derivatives of the same P

    if PP.P__ and not isinstance(PP.P__[0], list):  # PP could be seg, when fseg = true in agg_recursion

        PP.uplink_layers[-1] += [inp.uplink_.copy()]  # seg.link_s are all misses now
        PP.downlink_layers[-1] += [inp.downlink_.copy()]

        for P in inp.P__:  # add Ps in P__[y]:
            P.root = object  # reset root, to be assigned next sub_recursion
            PP.P__.append(P)
    else:
        for P in inp.P__:  # add Ps in P__[y]:
            if not PP.P__: PP.P__.append([P])
            else:
                current_ys = [P_[0].y for P_ in PP.P__]  # list of current-layer seg rows
                if P.y in current_ys:
                    PP.P__[current_ys.index(P.y)].append(P)  # append P row
                elif P.y > current_ys[0]:  # P.y > largest y in ys
                    PP.P__.insert(0, [P])
                elif P.y < current_ys[-1]:  # P.y < smallest y in ys
                    PP.P__.append([P])
                elif P.y < current_ys[0] and P.y > current_ys[-1]:  # P.y in between largest and smallest value
                    PP.P__.insert(P.y - current_ys[-1], [P])

            # add seg links: we may need links of all terminated segs, for rng+
            for derP in inp.P__[0].downlink_layers[-1]:  # if downlink not in current PP's downlink and not part of the seg in current PP:
                if derP not in PP.downlink_layers[-1] and derP.P.root not in PP.seg_levels[fPd][-1]:
                    PP.downlink_layers[-1] += [derP]
            for derP in inp.P__[-1].uplink_layers[-1]:  # if downlink not in current PP's downlink and not part of the seg in current PP:
                if derP not in PP.downlink_layers[-1] and derP.P.root not in PP.seg_levels[fPd][-1]:
                    PP.uplink_layers[-1] += [derP]

def accum_layer(top_layer, der_layer):

    for i, (_param, param) in enumerate(zip(top_layer, der_layer)):  # include all summable derP variables into params?
        if isinstance(_param, tuple):
            if len(_param) == 2:  # (sin_da, cos_da)
                _sin_da, _cos_da = _param
                sin_da, cos_da = param
                sum_sin_da = (cos_da * _sin_da) + (sin_da * _cos_da)  # sin(α + β) = sin α cos β + cos α sin β
                sum_cos_da = (cos_da * _cos_da) - (sin_da * _sin_da)  # cos(α + β) = cos α cos β - sin α sin β
                top_layer[i] = (sum_sin_da, sum_cos_da)
            else:  # (sin_da0, cos_da0, sin_da1, cos_da1)
                _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _param
                sin_da0, cos_da0, sin_da1, cos_da1 = param
                sum_sin_da0 = (cos_da0 * _sin_da0) + (sin_da0 * _cos_da0)  # sin(α + β) = sin α cos β + cos α sin β
                sum_cos_da0 = (cos_da0 * _cos_da0) - (sin_da0 * _sin_da0)  # cos(α + β) = cos α cos β - sin α sin β
                sum_sin_da1 = (cos_da1 * _sin_da1) + (sin_da1 * _cos_da1)
                sum_cos_da1 = (cos_da1 * _cos_da1) - (sin_da1 * _sin_da1)
                top_layer[i] = (sum_sin_da0, sum_cos_da0, sum_sin_da1, sum_cos_da1)
        else:  # scalar
            top_layer[i] += param


def form_PP_root(seg_t, root_rdn):  # form PPs from linked segs

    PP_t = []
    for fPd in 0, 1:
        PP_ = []
        seg_ = seg_t[fPd]
        for seg in seg_:  # bottom-up
            if not isinstance(seg.root, CPP):  # seg is not already in PP initiated by some prior seg
                # link CderPs: P.root=seg, _P.root=_seg
                if seg.P__[-1].uplink_layers[-1] or seg.P__[0].downlink_layers[-1]:
                    form_PP_(PP_, [seg], seg.P__[-1].uplink_layers[-1].copy(), seg.P__[0].downlink_layers[-1].copy(), fPd)
                else:
                    PP_ += [sum2PP([seg], fPd)]  # single-seg PP

        PP_t.append(PP_)  # PP_segs are converted to PPs in sum2PP and form_PP_

    return PP_t  # PPm_, PPd_

# under review, miss_uuplink_, miss_ddownlink_ from top/bottom segs only, use in sum2PP?

def form_PP_(PP_, PP_segs, uplink_, downlink_, fPd):

    # flood-fill PP_segs with vertically linked same-sign segments:
    match_uuplink_, match_ddownlink_  = [],[]
    '''
    all local link_s are concatenated from exposed PP_segs
    miss_uplink_, miss_downlink_ are now in seg.links
    '''
    # scan_link_(): separate upward and downward recursion?
    
    # 2 steps process, pack all PP_segs first, then search for the next level links
    upderP_ = []
    for upderP in uplink_:
        if isinstance(upderP._P.root, CPP):
            PP_segs += [upderP._P.root] 
            upderP_.append(upderP)
                     
    downderP_ = []
    for downderP in downlink_:
        if isinstance(downderP.P.root, CPP):  # lowest row doesn't have P's root
            PP_segs += [downderP.P.root]  # add next row of segs in matching downlinks:      
            downderP_.append(downderP)
    
    for upderP in upderP_:
        for uupderP in upderP._P.uplink_layers[-1]:
            # check match_uuplink_+miss_uplink_?
            if uupderP not in match_uuplink_ and uupderP._P.root not in PP_segs:
                match_uuplink_ += [uupderP]

    for downderP in downderP_:
        for ddownderP in downderP.P.downlink_layers[0]:
            if ddownderP not in match_ddownlink_ and ddownderP.P.root not in PP_segs:
                match_ddownlink_ += [ddownderP]


    if match_uuplink_ or match_ddownlink_:  # recursive compare sign of next-layer uplinks
        form_PP_(PP_, PP_segs, match_uuplink_, match_ddownlink_, fPd)
    else:
        PP_ += [sum2PP(PP_segs, fPd)]  # PP_segs is converted to PP
        '''
        PP is a graph with segs as 1D "vertices", each with two sets of edges or branching points: seg.uplink_ and seg.downlink_.
        Each edge is CderP, with derP.rdn *= len(P.uplink_|_P.downlink_)
        '''
    return PP_


# yet to be updated
def merge_PP(_PP, PP, fPd):  # only for PP splicing

    for seg in PP.seg_levels[fPd][-1]:  # merge PP_segs into _PP:
        accum_CPP(_PP, seg, fPd)
        _PP.seg_levels[fPd][-1] += [seg]

    # merge uplinks and downlinks
    for uplink in PP.uplink_layers:
        if uplink not in _PP.uplink_layers:
            _PP.uplink_layers += [uplink]
    for downlink in PP.downlink_layers:
        if downlink not in _PP.downlink_layers:
            _PP.downlink_layers += [downlink]


def sub_recursion(root_layers, PP_, frng):  # compares param_layers of derPs in generic PP, form or accum top derivatives

    comb_layers = []
    for PP in PP_:  # PP is generic higher-composition pattern, P is generic lower-composition pattern
                    # both P and PP may be recursively formed higher-derivation derP and derPP, etc.
        if frng: PP_V = PP.params[-1][0] - ave_mPP * PP.rdn; rng = PP.rng+1; min_L = rng * 2  # V: value of sub_recursion per PP
        else:    PP_V = PP.params[-1][1] - ave_dPP * PP.rdn; rng = PP.rng; min_L = 3  # need 3 Ps to compute layer2, etc.
        if PP_V > 0 and PP.nderP > min_L:

            PP.rdn += 1  # rdn to prior derivation layers
            PP.rng = rng
            Pm__ = comp_P_rng(PP.P__, rng)
            Pd__ = comp_P_der(PP.P__)
            # reversed P__ because form_seg_root will reverse them again
            sub_segm_ = form_seg_root([Pm_ for Pm_ in reversed(Pm__)], root_rdn=PP.rdn, fPd=0)
            sub_segd_ = form_seg_root([Pd_ for Pd_ in reversed(Pd__)], root_rdn=PP.rdn, fPd=1)
            sub_PPm_, sub_PPd_ = form_PP_root((sub_segm_, sub_segd_), root_rdn=PP.rdn)  # forms PPs: parameterized graphs of linked segs

            PP.layers = [(sub_PPm_, sub_PPd_)]
            if sub_PPm_:   # or rng+n to reduce clustering costs?
                sub_recursion(PP.layers, sub_PPm_, frng=1)  # rng+ comp_P in PPms, form param_layer, sub_PPs
            if sub_PPd_:
                sub_recursion(PP.layers, sub_PPd_, frng=0)  # der+ comp_P in PPds, form param_layer, sub_PPs

            if PP.layers:  # pack added sublayers:
                new_comb_layers = []
                for (comb_sub_PPm_, comb_sub_PPd_), (sub_PPm_, sub_PPd_) in zip_longest(comb_layers, PP.layers, fillvalue=([], [])):
                    comb_sub_PPm_ += sub_PPm_
                    comb_sub_PPd_ += sub_PPd_
                    new_comb_layers.append((comb_sub_PPm_, comb_sub_PPd_))  # add sublayer
                comb_layers = new_comb_layers

    if comb_layers: root_layers += comb_layers


def agg_recursion(blob, fseg):  # compositional recursion per blob.Plevel. P, PP, PPP are relative terms, each may be of any composition order


    if fseg: PP_t = [blob.seg_levels[0][-1], blob.seg_levels[1][-1]]   # blob is actually PP, recursion forms segP_t, seg_PP_t, etc.
    else: PP_t = [blob.levels[0][-1], blob.levels[1][-1]]  # input-level composition Ps, initially PPs
    PPP_t = []  # next-level composition Ps, initially PPPs  # for fiPd, PP_ in enumerate(PP_t): fiPd = fiPd % 2  # dir_blob.M += PP.M += derP.m

    n_extended = 0
    for i, PP_ in enumerate(PP_t):   # fiPd = fiPd % 2

        # splicing segs when fseg = true, seg__ is 2D: cross-sign (same-sign), converted to PP_ and PP respectively
        # or splicing PPs when fseg = false
        if fseg:
            splice_segs(PP_)
        else:
            splice_PPs(PP_, frng=1-i)

        fiPd = i % 2
        if fiPd: ave_PP = ave_dPP
        else:    ave_PP = ave_mPP

        if fseg: M = ave- np.hypot(blob.params[0][5], blob.params[0][6])  # hypot(dy, dx)
        else: M = ave-abs(blob.G)
        if M > ave_PP * blob.rdn and len(PP_)>1:  # >=2 comparands
            n_extended += 1

            PPm__ = [comp_aggP_root(deepcopy(PP_))]  # PP is generic for lower-level composition
            PPd__ = [comp_aggP_root(deepcopy(PP_))]
            # sections below need further update
            segm_ = form_seg_root(PPm__, root_rdn=2, fPd=0)  # forms segments: parameterized stacks of (P,derP)s
            segd_ = form_seg_root(PPd__, root_rdn=2, fPd=1)  # seg is a stack of (P,derP)s

            PPPm_, PPPd_ = form_PP_root([segm_, segd_], root_rdn=2)  # PPP is generic next-level composition

            splice_PPs(PPPm_, frng=1)
            splice_PPs(PPPd_, frng=0)
            PPP_t += [PPPm_, PPPd_]  # flat version

            if PPPm_: sub_recursion([], PPPm_, frng=1)  # rng+
            if PPPd_: sub_recursion([], PPPd_, frng=0)  # der+
        else:
            PPP_t += [[], []]  # replace with neg PPPs?

    if fseg:
        blob.seg_levels += [PPP_t]  # new level of segPs
    else:
        blob.levels.append(PPP_t)  # levels of dir_blob are Plevels

    if n_extended/len(PP_t) > 0.5:  # mean ratio of extended PPs
        agg_recursion(blob, fseg)


def comp_aggP_root(PP_):

    uplink_layers = [[] for PP in PP_]
    downlink_layers = deepcopy(uplink_layers)

    for i, PP in enumerate(PP_):
        for _PP in PP.uplink_layers[-1]:

            derPP = comp_derP(_PP, PP)  # cross-sign if PPd?
            uplink_layers[i] += [derPP]  # add derPP
            if _PP in PP_: downlink_layers[PP_.index(_PP)] += [derPP]

    # update links
    for PP, uplink_layer, downlink_layer in zip_longest(PP_, uplink_layers, downlink_layers, fillvalue=[]):
        PP.uplink_layers += [uplink_layer]
        PP.downlink_layers += [downlink_layer]

    return PP_

# pending update
def splice_segs(seg_):
    pass

# pending update
# draft , splice 2 PPs for now
def splice_PPs(PP_, frng):  # splice select PP pairs if der+ or triplets if rng+

    spliced_PP_ = []
    while PP_:
        _PP = PP_.pop(0)  # pop PP, so that we can differentiate between tested and untested PPs
        tested_segs = []  # we need this because we may add new seg during splicing process, and those new seg need to check their link for splicing too
        _segs = _PP.seg_levels[0]

        while _segs:
            _seg = _segs.pop(0)
            _avg_y = sum([P.y for P in _seg.P__])/len(_seg.P__)  # y centroid for _seg

            for link in _seg.uplink_layers[1] + _seg.downlink_layers[1]:
                seg = link.P.root  # missing link of current seg

                if seg.root is not _PP:  # this may occur after the merging where multiple links are having same PP
                    avg_y = sum([P.y for P in seg.P__])/len(seg.P__)  # y centroid for seg

                    # test for y distance (temporary)
                    if (_avg_y - avg_y) < ave_splice:
                        if seg.root in PP_: PP_.remove(seg.root)  # remove merged PP
                        elif seg.root in spliced_PP_: spliced_PP_.remove(seg.root)
                        # splice _seg's PP with seg's PP
                        merge_PP(_PP, seg.root)

            tested_segs += [_seg]  # pack tested _seg
        _PP.seg_levels[0] = tested_segs
        spliced_PP_ += [_PP]

    return spliced_PP_

def comp_dx(P):  # cross-comp of dx s in P.dert_

    Ddx = 0
    Mdx = 0
    dxdert_ = []
    _dx = P.dert_[0][2]  # first dx
    for dert in P.dert_[1:]:
        dx = dert[2]
        ddx = dx - _dx
        if dx > 0 == _dx > 0: mdx = min(dx, _dx)
        else: mdx = -min(abs(dx), abs(_dx))
        dxdert_.append((ddx, mdx))  # no dx: already in dert_
        Ddx += ddx  # P-wide cross-sign, P.L is too short to form sub_Ps
        Mdx += mdx
        _dx = dx
    P.dxdert_ = dxdert_
    P.Ddx = Ddx
    P.Mdx = Mdx


def comp_P(_P, P):  # forms vertical derivatives of params per P in _P.uplink, conditional ders from norm and DIV comp

    # compared P params:
    x, L, M, Ma, I, Dx, Dy, sin_da0, cos_da0, sin_da1, cos_da1 = P.params
    _x, _L, _M, _Ma, _I, _Dx, _Dy, _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _P.params

    dx = _x - x;  mx = ave_dx - abs(dx)  # mean x shift, if dx: rx = dx / ((L+_L)/2)? no overlap, offset = abs(x0 -_x0) + abs(xn -_xn)?
    dI = _I - I;  mI = ave_I - abs(dI)
    dM = _M - M;  mM = min(_M, M)
    dMa = _Ma - Ma;  mMa = min(_Ma, Ma)  # dG, dM are directional, re-direct by dx?
    dL = _L - L * np.hypot(dx, 1); mL = min(_L, L)  # if abs(dx) > ave: adjust L as local long axis, no change in G,M
    # G, Ga:
    G = np.hypot(Dy, Dx); _G = np.hypot(_Dy, _Dx)  # compared as scalars
    dG = _G - G;  mG = min(_G, G)
    Ga = (cos_da0 + 1) + (cos_da1 + 1); _Ga = (_cos_da0 + 1) + (_cos_da1 + 1)  # gradient of angle, +1 for all positives?
    # or Ga = np.hypot( np.arctan2(*Day), np.arctan2(*Dax)?
    dGa = _Ga - Ga;  mGa = min(_Ga, Ga)

    # comp angle:
    _sin = _Dy / (1 if _G==0 else _G); _cos = _Dx / (1 if _G==0 else _G)
    sin  = Dy / (1 if G==0 else G); cos = Dx / (1 if G==0 else G)
    sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
    dangle = np.arctan2(sin_da, cos_da)  # vertical difference between angles
    mangle = ave_dangle - abs(dangle)  # indirect match of angles, not redundant as summed

    # comp angle of angle: forms daa, not gaa?
    sin_dda0 = (cos_da0 * _sin_da0) - (sin_da0 * _cos_da0)
    cos_dda0 = (cos_da0 * _cos_da0) + (sin_da0 * _sin_da0)
    sin_dda1 = (cos_da1 * _sin_da1) - (sin_da1 * _cos_da1)
    cos_dda1 = (cos_da1 * _cos_da1) + (sin_da1 * _sin_da1)

    daangle = (sin_dda0, cos_dda0, sin_dda1, cos_dda1)
    # day = [-sin_dda0 - sin_dda1, cos_dda0 + cos_dda1]
    # dax = [-sin_dda0 + sin_dda1, cos_dda0 + cos_dda1]
    gay = np.arctan2( (-sin_dda0 - sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in y?
    gax = np.arctan2( (-sin_dda0 + sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in x?
    daangle = np.arctan2( gay, gax)  # probably wrong
    maangle = ave_daangle - abs(daangle)  # match between aangles, not redundant as summed

    dP = abs(dx)-ave_dx + abs(dI)-ave_I + abs(G)-ave_G + abs(Ga)-ave_Ga + abs(dM)-ave_M + abs(dMa)-ave_Ma + abs(dL)-ave_L
    # sum to evaluate for der+, abs diffs are distinct from directly defined matches:
    mP = mx + mI + mG + mGa + mM + mMa + mL + mangle + maangle

    params = [mP, dP, dx, mx, dL, mL, dI, mI, dG, mG, dGa, mGa, dM, mM, dMa, mMa, dangle, mangle, daangle, maangle]
    # or summable params only, all Gs are computed at termination?

    x0 = min(_P.x0, P.x0)
    xn = max(_P.x0+_P.L, P.x0+P.L)
    L = xn-x0

    return CderP(x0=x0, L=L, y=_P.y, params=params, P=P, _P=_P)


def comp_derP(_derP, derP):

    nparams = len(_derP.params)
    derivatives = []
    hyps = []
    mP = 0  # for rng+ eval
    dP = 0  # for der+ eval

    for i, (_param, param) in enumerate(zip(_derP.params, derP.params)):
        # get param type:
        param_type = int(i/ (2 ** (nparams-1)))  # for 9 compared params, but there are more in higher layers?

        if param_type == 0:  # x
            _x = param; x = param
            dx = _x - x; mx = ave_dx - abs(dx)
            derivatives.append(dx); derivatives.append(mx)
            hyps.append(np.hypot(dx, 1))
            dP += dx; mP += mx

        elif param_type == 1:  # I
            _I = _param; I = param
            dI = _I - I; mI = ave_I - abs(dI)
            derivatives.append(dI); derivatives.append(mI)
            dP += dI; mP += mI

        elif param_type == 2:  # G
            hyp = hyps[i%param_type]
            _G = _param; G = param
            dG = _G - G/hyp;  mG = min(_G, G)  # if comp_norm: reduce by hypot
            derivatives.append(dG); derivatives.append(mG)
            dP += dG; mP += mG

        elif param_type == 3:  # Ga
            _Ga = _param; Ga = param
            dGa = _Ga - Ga;  mGa = min(_Ga, Ga)
            derivatives.append(dGa); derivatives.append(mGa)
            dP += dGa; mP += mGa

        elif param_type == 4:  # M
            hyp = hyps[i%param_type]
            _M = _param; M = param
            dM = _M - M/hyp;  mM = min(_M, M)
            derivatives.append(dM); derivatives.append(mM)
            dP += dM; mP += mM

        elif param_type == 5:  # Ma
            _Ma = _param; Ma = param
            dMa = _Ma - Ma;  mMa = min(_Ma, Ma)
            derivatives.append(dMa); derivatives.append(mMa)
            dP += dMa; mP += mMa

        elif param_type == 6:  # L
            hyp = hyps[i%param_type]
            _L = _param; L = param
            dL = _L - L/hyp;  mL = min(_L, L)
            derivatives.append(dL); derivatives.append(mL)
            dP += dL; mP += mL

        elif param_type == 7:  # angle, (sin_da, cos_da)
            if isinstance(_param, tuple):  # (sin_da, cos_da)
                 _sin_da, _cos_da = _param; sin_da, cos_da = param
                 sin_dda = (cos_da * _sin_da) - (sin_da * _cos_da)  # sin(α - β) = sin α cos β - cos α sin β
                 cos_dda = (cos_da * _cos_da) + (sin_da * _sin_da)  # cos(α - β) = cos α cos β + sin α sin β
                 dangle = (sin_dda, cos_dda)  # da
                 mangle = ave_dangle - abs(np.arctan2(sin_dda, cos_dda))  # ma is indirect match
                 derivatives.append(dangle); derivatives.append(mangle)
                 dP += np.arctan2(sin_dda, cos_dda); mP += mangle
            else: # m or scalar
                _mangle = _param; mangle = param
                dmangle = _mangle - mangle;  mmangle = min(_mangle, mangle)
                derivatives.append(dmangle); derivatives.append(mmangle)
                dP += dmangle; mP += mmangle

        elif param_type == 8:  # dangle   (sin_da0, cos_da0, sin_da1, cos_da1)
            if isinstance(_param, tuple):  # (sin_da, cos_da)
                _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _param
                sin_da0, cos_da0, sin_da1, cos_da1 = param

                sin_dda0 = (cos_da0 * _sin_da0) - (sin_da0 * _cos_da0)
                cos_dda0 = (cos_da0 * _cos_da0) + (sin_da0 * _sin_da0)
                sin_dda1 = (cos_da1 * _sin_da1) - (sin_da1 * _cos_da1)
                cos_dda1 = (cos_da1 * _cos_da1) + (sin_da1 * _sin_da1)
                daangle = (sin_dda0, cos_dda0, sin_dda1, cos_dda1)
                # day = [-sin_dda0 - sin_dda1, cos_dda0 + cos_dda1]
                # dax = [-sin_dda0 + sin_dda1, cos_dda0 + cos_dda1]
                gay = np.arctan2( (-sin_dda0 - sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in y?
                gax = np.arctan2( (-sin_dda0 + sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in x?
                maangle = ave_dangle - abs(np.arctan2(gay, gax))  # match between aangles, probably wrong
                derivatives.append(daangle); derivatives.append(maangle)
                dP += daangle; mP += maangle

            else:  # m or scalar
                _maangle = _param; maangle = param
                dmaangle = _maangle - maangle;  mmaangle = min(_maangle, maangle)
                derivatives.append(dmaangle); derivatives.append(mmaangle)
                dP += dmaangle; mP += mmaangle

    x0 = min(_derP.x0, derP.x0)
    xn = max(_derP.x0+_derP.L, derP.x0+derP.L)
    L = xn-x0

    dderP = CderP(x0=x0, L=L, y=_derP.y, params=derivatives, P=derP, _P=_derP)

    return dderP


# old draft
def splice_dir_blob_(dir_blobs):

    for i, _dir_blob in enumerate(dir_blobs):
        for fPd in 0, 1:
            PP_ = _dir_blob.levels[0][fPd]

            if fPd: PP_val = sum([PP.mP for PP in PP_])
            else:   PP_val = sum([PP.dP for PP in PP_])

            if PP_val - ave_splice > 0:  # high mPP pr dPP

                _top_P_ = _dir_blob.P__[0]
                _bottom_P_ = _dir_blob.P__[-1]

                for j, dir_blob in enumerate(dir_blobs):
                    if _dir_blob is not dir_blob:

                        top_P_ = dir_blob.P__[0]
                        bottom_P_ = dir_blob.P__[-1]
                        # test y adjacency
                        if (_top_P_[0].y-1 == bottom_P_[0].y) or (top_P_[0].y-1 == _bottom_P_[0].y):
                            # tet x overlap
                             _x0 = min([_P.x0 for _P_ in _dir_blob.P__ for _P in _P_])
                             _xn = min([_P.x0+_P.L for _P_ in _dir_blob.P__ for _P in _P_])
                             x0 = min([P.x0 for P_ in dir_blob.P__ for P in P_])
                             xn = min([P.x0+_P.L for P_ in dir_blob.P__ for P in P_])
                             if (x0 - 1 < _xn and xn + 1 > _x0) or  (_x0 - 1 < xn and _xn + 1 > x0) :
                                 splice_dir_blobs(_dir_blob, dir_blob)  # splice dir_blob into _dir_blob
                                 dir_blobs[j] = _dir_blob

def splice_dir_blobs(_blob, blob):
    # merge blob into _blob here
    pass


# draw segments within single dir_blob
def draw_seg_(dir_blob, seg_):
    import random
    import cv2
    import os

    x0 = min([P.x0 for seg in seg_ for P in seg.P__])
    xn = max([P.x0 + P.L for seg in seg_ for P in seg.P__])
    y0 = min([P.y for seg in seg_ for P in seg.P__])
    yn = max([P.y for seg in seg_ for P in seg.P__])

    img = np.zeros((yn - y0 + 1, xn - x0 + 1, 3), dtype="uint8")

    for seg in seg_:
        current_colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        for P in seg.P__:
            img[P.y - y0, P.x0 - x0:P.x0 - x0 + P.L] = current_colour

    cv2.imwrite(os.getcwd() + "/images/comp_slice/img_" + str(dir_blob.id) + ".png", img)

# draw segments within single PP
def draw_PP_segs(dir_blob, PP_):
    import random
    import cv2
    import os

    x0 = min([P.x0 for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])
    xn = max([P.x0 + P.L for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])
    y0 = min([P.y for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])
    yn = max([P.y for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])

    for PP in PP_:
        img = np.zeros((yn - y0 + 1, xn - x0 + 1, 3), dtype="uint8")
        for seg in PP.seg_levels[0]:
            current_colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            for P in seg.P__:
                img[P.y - y0, P.x0 - x0:P.x0 - x0 + P.L] = current_colour

        cv2.imwrite(os.getcwd() + "/images/comp_slice/img_" + str(dir_blob.id) + "_PP_"+str(PP.id)+".png", img)


# draw PPs within single dir_blob
def draw_PPs(dir_blob, PP_, fspliced):
    import random
    import cv2
    import os

    x0 = min([P.x0 for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])
    xn = max([P.x0 + P.L for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])
    y0 = min([P.y for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])
    yn = max([P.y for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])

    img = np.zeros((yn - y0 + 1, xn - x0 + 1, 3), dtype="uint8")
    for PP in PP_:
        current_colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for seg in PP.seg_levels[0]:
            for P in seg.P__:
                img[P.y - y0, P.x0 - x0:P.x0 - x0 + P.L] = current_colour

    if fspliced:
        cv2.imwrite(os.getcwd() + "/images/comp_slice/img_" + str(dir_blob.id) + "_PP.png", img)
    else:
        cv2.imwrite(os.getcwd() + "/images/comp_slice/img_" + str(dir_blob.id) + "_PP_spliced.png", img)