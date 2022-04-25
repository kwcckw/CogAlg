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
from copy import deepcopy
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
    # if comp_dx:
    Mdx = int
    Ddx = int
    # composite params:
    dert_ = list  # array of pixel-level derts, redundant to upconnect_, only per blob?
    upconnect_ = list
    downconnect_ = list
    # only in Pd:
    Pm = object  # reference to root P
    dxdert_ = list
    # only in Pm:
    Pd_ = list

class CderP(ClusterStructure):  # tuple of derivatives in P upconnect_ or downconnect_

    dP = int
    mP = int
    params = list  # P derivation layer, n_params = 9 * 2**der_cnt, flat, decoded by mapping each m,d to lower-layer param
    x0 = int  # redundant to params:
    x = float  # median x
    L = int
    sign = NoneType  # g-ave + ave-ga sign
    y = int  # for vertical gaps in PP.P__, replace with derP.P.y?
    P = object  # lower comparand
    _P = object  # higher comparand
    root = object  # segment, contains this derP
    # higher derivatives
    rdn = int  # mrdn, + uprdn if branch overlap?
    upconnect_ = list  # tuples of higher-row higher-order derivatives per derP
    downconnect_ = list
   # from comp_dx
    fdx = NoneType

class CPP(CP, CderP):  # derP params are inherited from P

    params = list  # derivation layer += derP params, param L is actually Area
    sign = bool
    rng = lambda: 1  # rng starts with 1
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs
    Rdn = int  # for accumulation only
    nderP = int  # len 2D derP__ in levels[0][fPd]?  ly = len(derP__), also x, y?
    upconnect_ = list
    downconnect_ = list
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    box = list  # for visualization only, original box before flipping
    mask__ = bool
    P__ = list
    derP__ = list  # input
    seg_levels = list  # from 1st agg_recursion, seg_t = levels[0], segP_t = levels[n], seg: stack of Ps
    PPP_levels = list  # from 2nd agg_recursion, PP_t = levels[0], from form_PP, before recursion
    layers = list  # from sub_recursion, each is derP_t
    root = object  # higher-order segP or PPP

# Functions:

def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    segment_by_direction(blob, verbose=False)  # forms blob.dir_blobs
    for dir_blob in blob.dir_blobs:  # dir_blob should be CBlob

        P__ = slice_blob(dir_blob, verbose=False)  # cluster dir_blob.dert__ into 2D array of blob slices
        # comp_dx_blob(P__), comp_dx?
        derP__ = comp_P_root(P__, rng=1, frng=0)  # scan_P_, comp_P, or comp_layers if called from sub_recursion

        seg_t = form_seg_root(derP__, root_rdn=2)
        PPm_, PPd_ = form_PP_root(seg_t, root_rdn=2)  # forms segments: stacks of (P,derP)s, combines them into PPs

#        splice_PPs(PPm_, frng=1)  # splicing segs, seg__ is 2D: cross-sign (same-sign), converted to PP_ and PP respectively
#        splice_PPs(PPd_, frng=0)
#        sub_recursion([], PPm_, frng=1)  # rng+ comp_P in PPms, -> param_layer, form sub_PPs
#        sub_recursion([], PPd_, frng=0)  # der+ comp_P in PPds, -> param_layer, form sub_PPs
#
#        for PP_ in (PPm_, PPd_):  # 1st agglomerative recursion is per PP, appending PP.seg_levels, not blob.levels:
#            for PP in PP_:
#                agg_recursion(PP, fseg=1)  # higher-composition comp_seg -> segPs.. per seg__[n], in PP.seg_levels
#        dir_blob.levels = [(PPm_, PPd_)]
#        agg_recursion(dir_blob, fseg=0)  # 2nd call per dir_blob.PP_s formed in 1st call, forms PPP..s and dir_blob.levels
#
#    splice_dir_blob_(blob.dir_blobs)


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
                    Pdert_ = []
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

def comp_P_root(P__, rng, frng):  # vertically compares y-adjacent and x-overlapping Ps: blob slices, forming 2D derP__

    # if der+: P__ is last-call derP__, derP__=[], form new derP__
    # if rng+: P__ is last-call P__, accumulate derP__ with new_derP__
    # 2D array of derivative tuples from P__[n], P__[n-rng], sub-recursive:
    derP__ = []
    for P_ in P__:
        for P in P_:
            P.upconnect_, P.downconnect_ = [],[]  # reset connects and PP refs in the last layer only
            if isinstance(P, CderP): P.root = None

    for i, _P_ in enumerate(P__):  # higher compared row
        derP_ = []
        if i+rng < len(P__):  # rng=1 unless rng+ fork
            P_ = P__[i+rng]   # lower compared row
            for P in P_:
                if frng: cP = P.P  # rng+, compared P is lower derivation
                else:    cP = P  # der+, compared P is top derivation
                for _P in _P_:  # upper compared row
                    if frng: _cP = _P.P
                    else:    _cP = _P
                    # form sub_Pds for comp_dx?
                    # test for x overlap between P and _P in 8 directions, all Ps are from +ve derts:
                    if (cP.x0 - 1 < (_cP.x0 + _cP.L) and (cP.x0 + cP.L) + 1 > _cP.x0):
                        if isinstance(cP, CPP) or isinstance(cP, CderP):
                            derP = comp_layer(_cP, cP)  # form vertical derivatives of horizontal P params
                        else:
                            derP = comp_P(_cP, cP)  # form higher vertical derivatives of derP or PP params
                        derP.y=P.y
                        if frng:  # accumulate derP through rng+ recursion:
                            accum_layer(derP.params, P.params)
                        if not P.downconnect_:  # initial row per root PP, then follow upconnect_
                            derP_.append(derP)
                        P.upconnect_.append(derP)  # per P for form_PP
                        _P.downconnect_.append(derP)

                    elif (cP.x0 + cP.L) < _cP.x0:  # no P xn overlap, stop scanning lower P_
                        break
        if derP_:
            derP__ += [derP_]  # rows in blob or PP
        _P_ = P_

    return derP__


def form_seg_root(derP__, root_rdn):  # form segs from derPs

    seg_t = []
    for fPd in 0, 1:
        seg_ = []
        # we need deepcopy here, since we have fPd loop above 
        for derP_ in deepcopy(derP__):  # get a row of derPs, bottom-up
            for derP in derP_:
                if fPd: derP.rdn = (derP.mP > derP.dP); derP.sign = derP.dP >= ave_dP * derP.rdn
                else:   derP.rdn = (derP.dP >= derP.mP); derP.sign = derP.mP > ave_mP * derP.rdn

                if derP._P.upconnect_:  # in form_seg_: if len matching_upconnect_==1 and len matching_upconnect_[0].matching_downconnect_==1
                    form_seg_(seg_, [derP], fPd)  # accum seg with 1/1 matching connects
                    # also accumulate PP_missing_upconnect_ and PP_missing_downconnect_ per derP?
                else:
                    seg_.append( sum2seg([derP]) )  # no upconnect_, immediate termination
                    derP.P, derP._P = seg_[-1], object  # update P to seg, _P to empty object due to no upconnect
                    
        seg_t.append(seg_)

    return seg_t  # segm_, segd_


def form_seg_(seg_, seg_derPs, fPd):  # form same-sign vertically contiguous segments

    matching_upconnect_ = []
    for derP in seg_derPs[-1]._P.upconnect_:  # top derP upconnect, seg_derPs is not converted to CPP yet

        if fPd: derP.rdn = (derP.mP > derP.dP); derP.sign = derP.dP >= ave_dP * derP.rdn
        else:   derP.rdn = (derP.dP >= derP.mP); derP.sign = derP.mP > ave_mP * derP.rdn
        if derP.sign == seg_derPs[0].sign:
            matching_upconnect_ += [derP]
        # else: missing_upconnect_ += [derP]  # add to PP_missing_upconnect_ at seg termination, same for missing_downconnect_?

    if len(matching_upconnect_) > 1:
        seg = sum2seg(seg_derPs)  # convert seg_derPs to seg
        for derP in matching_upconnect_:
            # derP._P.downconnect_ += [derP]  # add to downconnect_ per upconnect (this should be already added in comp_P_root, but not for form_PP_) 
            derP.P = seg  # downconnected seg in upconnect derP 
            seg.upconnect_ += [derP]
        seg.upconnect_ = matching_upconnect_
        for derP in seg.downconnect_: derP._P = seg  # upconnected seg
        seg_.append(seg)
    else:
        if matching_upconnect_ and len(matching_upconnect_[0].downconnect_)==1:  # 1/1 matching connects per derP
            seg_derPs += [derP]
            if seg_derPs[-1]._P.upconnect_:
                form_seg_(seg_, seg_derPs, fPd)  # recursive compare sign of next-layer upconnects
        else:
            seg_.append( sum2seg(seg_derPs))  # terminate at 0 matching upconnect
'''
        generic rng+|der+ rdn:
        if fPd: inp.rdn += (inp.mP > inp.dP)  # PPd / vD sign, distinct from directly defined match:
        else:   inp.rdn += (inp.dP >= inp.mP)
        # if branch rdn: inp.rdn += sum([1 for upderP in derP.P.upconnect_ if upderP.dP >= derP.dP])
'''

def form_PP_root(seg_t, root_rdn):  # form segs from derPs, then PPs from segs

    PP_t = []
    for fPd in 0, 1:
        PP_segs_ = []
        seg_ = seg_t[fPd]
        for seg in seg_:  # bottom-up

            if seg.upconnect_:  # seg.upconnect is CderP with P=seg and _P=_seg
                form_PP_(PP_segs_, [seg], seg.upconnect_, fPd)
            else:
                sum2PP(PP_segs_, [seg])  # single-seg PP

        PP_t.append(PP_segs_)  # PP_segs are replaced with PPs in sum2PP and form_PP_
        '''
        PP is a graph with segs as edges and multiple connects (branching points) per seg as vertex / node.
        Vertex is seg.upconnect_| downconnect_: 0-n derPs, derP.rdn *= len(P.upconnect_| _P.downconnect_)
        Vertices are always up or down, 1-to-many. Many-to-many connection consists of multiple vertices.
        '''
    return PP_t  # PPm_, PPd_


def form_PP_(PP_segs_, PP_segs, upconnect_, fPd):  # form PP of same-sign connected segments

    matching_upconnect_ = []
    missing_upconnect_ = []
    for derP in upconnect_:  # seg upconnects are derPs

        seg = derP._P
        if isinstance(seg, CPP):  # could be object from line 248 above
            if fPd: seg.rdn = (seg.mP > seg.dP); sign = seg.dP >= ave_dP * seg.rdn
            else:   seg.rdn = (seg.dP >= seg.mP); sign = seg.mP > ave_mP * seg.rdn
            if sign == PP_segs[0].sign:
                if seg not in matching_upconnect_: matching_upconnect_ += [seg]
            else:
                if seg not in missing_upconnect_: missing_upconnect_ += [seg]

    if not matching_upconnect_:
        sum2PP(PP_segs_, PP_segs)  # form PP
        PP = PP_segs_[-1]
        PP.upconnect_ = matching_upconnect_
        for seg in missing_upconnect_:
            seg.root.downconnect_ += [PP_segs[0]]  # upconnect's PP's downconnect should be current PP top row of seg
        for seg in PP.downconnect_: seg.upconnect_ += [PP]  # upconnected PP
    else:
        # not reviewed:
        PP_segs += matching_upconnect_
        # get upconnects of matching upconnects:
        _upconnect_ = [derP for upseg in matching_upconnect_ for derP in upseg.upconnect_ if isinstance(derP._P, CPP)]
        if _upconnect_:
            form_PP_(PP_segs_, PP_segs, _upconnect_, fPd)  # recursive compare sign of next-layer upconnects


def sum2seg(seg_derPs):  # sum params: merge vertically connected derPs into segment

    seg = CPP(x0=seg_derPs[0].x0, sign=seg_derPs[0].sign)

    for derP in seg_derPs:
        if not seg.params:
            seg.params = derP.params.copy()
        else:
            accum_layer(seg.params, derP.params)
        seg.x0 = min(seg.x0, derP.x0)
        seg.nderP += 1
        seg.mP += derP.mP
        seg.dP += derP.dP
        seg.Rdn += derP.rdn
        seg.y = max(seg.y, derP.y)  # or pass local y arg instead of derP.y?
        '''
        PP.rdn += root_rdn + PP.Rdn / PP.nderP  # PP rdn is recursion rdn + average (forks + upconnects) rdn
        '''
        # 1 derP or 1 P per line
        seg.derP__.insert(0, [derP])
        seg.P__.insert(0, [derP.P])
        seg.L = len(seg.derP__)  # seg.L is Ly
        derP.root = seg
        
        for down_derP in derP.downconnect_:
            # check if downconnect not in existing downconnect_ or downconnect not in seg_derPs
            if down_derP not in seg.downconnect_ and down_derP not in seg_derPs:
                seg.downconnect_ += [down_derP]
    return seg


def comp_P(_P, P):  # forms vertical derivatives of params per P in _P.upconnect, conditional ders from norm and DIV comp

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

    params = [dx, mx, dL, mL, dI, mI, dG, mG, dGa, mGa, dM, mM, dMa, mMa, dangle, mangle, daangle, maangle]
    # or summable params only, all Gs are computed at termination?

    x0 = min(_P.x0, P.x0)
    xn = max(_P.x0+_P.L, P.x0+P.L)
    L = xn-x0

    derP = CderP(x0=x0, L=L, y=_P.y, mP=mP, dP=dP, params=params, P=P, _P=_P)
    return derP


def sum2PP(PP_, PP_segs):  # sum params: derPs into segment or segs into PP

    PP = CPP(x0=PP_segs[0].x0, sign=PP_segs[0].sign, seg_levels = [[]])

    for seg in PP_segs:
        if isinstance(seg.root, CPP) and seg.root is not PP :  # inp.root may == PP if PP is we get PP from upconnect
            merge_PP(PP, seg.root, PP_)
        elif seg not in PP.seg_levels[0]:
            if not PP.params:
                PP.params = seg.params.copy()
            else:
                accum_layer(PP.params, seg.params)
            PP.x0 = min(PP.x0, seg.x0)
            PP.nderP += 1
            PP.mP += seg.mP
            PP.dP += seg.dP
            PP.Rdn += seg.rdn
            PP.y = max(seg.y, PP.y)  # or pass local y arg instead of derP.y?
            '''
            PP.rdn += root_rdn + PP.Rdn / PP.nderP  # PP rdn is recursion rdn + average (forks + upconnects) rdn
            '''
            PP.seg_levels[0] += [seg]  # should be PP.seg_levels[0][fPd]?
            PP.L = len(PP.seg_levels[0])  # PP.L is Ly
            seg.root = PP

        for down_derP in seg.downconnect_:
            down_seg = down_derP.P
            if down_seg not in PP.downconnect_ and down_seg not in PP_segs:
                PP.downconnect += [down_seg]

    PP_ += [PP]

# different segs may initiate PPs that are connected through their upconnect_s.

def merge_PP(_PP, PP,PP_):  # merge PP into _PP

    PP_segs = []
    for derP_ in PP.derP__:
        for derP in derP_:
            current_derP_ = [derP for derP_ in _PP.derP__ for derP in derP_]
            if derP not in current_derP_:
                derP.root = object  # reset to prevent merging
                PP_segs += [derP]
    if PP_segs: sum2PP(PP_, PP_segs)  # no merge derP.root

    for up_PP in PP.upconnect_:
        if up_PP not in _PP.upconnect_:  # PP may have multiple downconnects
            _PP.upconnect_.append(up_PP)


    # no downconnect in current scheme, downconnect in comp_P_root only
    '''
    for i, down_PP in enumerate(PP.downconnect_):
        if PP in down_PP.upconnect_:
            down_PP.upconnect_[down_PP.upconnect_.index(PP)] = _PP  # update lower PP's upconnect from PP to _PP
            if down_PP not in _PP.downconnect_:
                _PP.downconnect_ += [down_PP]
    for segment in PP.segments:  # add segments from PP
        _PP.segments += [segment]
    '''

    if PP in PP_:
        PP_.remove(PP)  # merged PP


def accum_layer(top_layer, der_layer):

    for i, (_param, param) in enumerate(zip(top_layer, der_layer)):
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


# pending update for segments
def sub_recursion(root_sublayers, PP_, frng):  # compares param_layers of derPs in generic PP, form or accum top derivatives

    comb_sublayers = []
    for PP in PP_:  # PP is generic higher-composition pattern, P is generic lower-composition pattern
                    # both P and PP may be recursively formed higher-derivation derP and derPP, etc.

        if frng: PP_V = PP.mP - ave_mPP * PP.rdn; rng = PP.rng+1; min_L = rng * 2  # V: value of sub_recursion per PP
        else:    PP_V = PP.dP - ave_dPP * PP.rdn; rng = PP.rng; min_L = 3  # need 3 Ps to compute layer2, etc.
        if PP_V > 0 and PP.nderP > min_L:

            PP.rdn += 1  # rdn to prior derivation layers
            PP.rng = rng
            sub_derP__ = comp_P_root(PP.derP__, rng, frng)  # scan_P_, comp_P layer0;  splice PPs across dir_blobs?
            sub_PPm_, sub_PPd_ = form_PP_(sub_derP__, PP.rdn)  # each PP is a stack of (P, derP)s from comp_P

            PP.sublayers = [(sub_PPm_, sub_PPd_)]
            if sub_PPm_:   # or rng+n to reduce clustering costs?
                sub_recursion(PP.sublayers, sub_PPm_, frng=1)  # rng+ comp_P in PPms, form param_layer, sub_PPs
            if sub_PPd_:
                sub_recursion(PP.sublayers, sub_PPd_, frng=0)  # der+ comp_P in PPds, form param_layer, sub_PPs

            if PP.sublayers:  # pack added sublayers:
                new_comb_sublayers = []
                for (comb_sub_PPm_, comb_sub_PPd_), (sub_PPm_, sub_PPd_) in zip_longest(comb_sublayers, PP.sublayers, fillvalue=([], [])):
                    comb_sub_PPm_ += sub_PPm_
                    comb_sub_PPd_ += sub_PPd_
                    new_comb_sublayers.append((comb_sub_PPm_, comb_sub_PPd_))  # add sublayer
                comb_sublayers = new_comb_sublayers

    if comb_sublayers: root_sublayers += comb_sublayers


def agg_recursion(blob, fseg):  # compositional recursion per blob.Plevel. P, PP, PPP are relative terms, each may be of any composition order

    if fseg: PP_t = blob.seg_levels[-1]  # blob is actually PP, recursion forms segP_t, seg_PP_t, etc.
    else:    PP_t = blob.levels[-1]  # input-level composition Ps, initially PPs
    PPP_t = []  # next-level composition Ps, initially PPPs  # for fiPd, PP_ in enumerate(PP_t): fiPd = fiPd % 2  # dir_blob.M += PP.M += derP.m
    # below is not updated

    n_extended = 0
    for i, PP_ in enumerate(PP_t):   # fiPd = fiPd % 2
        fiPd = i % 2
        if fiPd: ave_PP = ave_dPP
        else:    ave_PP = ave_mPP

        M = ave-abs(blob.G)
        if M > ave_PP * blob.rdn and len(PP_)>1:  # >=2 comparands
            n_extended += 1

            derPP_ = comp_aggP_root(PP_, rng=1)  # PP is generic for lower-level composition
            PPPm_, PPPd_ = form_PP_(derPP_, root_rdn=2)  # PPP is generic next-level composition

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


def comp_aggP_root(PP_, rng):

    for PP in PP_: PP.downconnect_ = []  # new downconnect will be recomputed for derPP
    derPP__ = []

    for PP in PP_:
        for i, _PP in enumerate(PP.upconnect_):
            if isinstance(_PP, CPP):  # _PP could be replaced by derPP

                derPP = comp_layer(_PP, PP)  # cross-sign if PPd?
                PP.upconnect_[i] = derPP  # replace PP with derPP
                _PP.downconnect_ += [derPP]

                if not derPP__: derPP__.append([derPP])
                else:
                    # pack derPP in row at derPP.y:
                    current_ys = [derP_[0].P.y for derP_ in derPP__]  # list of current-layer derP rows
                    if derPP.P.y in current_ys:
                        derPP__[current_ys.index(derPP.P.y)].append(derPP)  # append derPP row
                    elif derPP.P.y > current_ys[-1]:  # derPP.y > largest y in ys
                        derPP__.append([derPP])
                    elif derPP.P.y < current_ys[0]:  # derPP.y < smallest y in ys
                        derPP__.insert(0, [derPP])
                    elif derPP.P.y > current_ys[0] and derPP.P.y < current_ys[-1] :  # derPP.y in between largest and smallest value
                        derPP__.insert(derPP.P.y-current_ys[0], [derPP])

    return derPP__

# draft, pending update for segments
def splice_PPs(PPP_, frng):  # splice select PP pairs if der+ or triplets if rng+

    for PPP in PPP_:
        if frng:  # rng fork
            for __PP_ in PPP.P__:  # top-down
                for i, __PP in enumerate(__PP_):
                    spliced_downconnect_ = []
                    for j, _downconnect in enumerate(__PP.downconnect_):
                        if isinstance(_downconnect, CderP): _PP = _downconnect.PP  # get PP reference if downconnect is derP
                        else:                               _PP = _downconnect
                        fbreak = 0
                        for k, downconnect in enumerate(_PP.downconnect_):
                            if isinstance(downconnect, CderP): PP = downconnect.PP
                            else:                              PP = downconnect
                            if __PP is not _PP and __PP is not PP and _PP is not PP:
                                max_rng = max(__PP.rng, PP.rng)
                                if max_rng > _PP.L:
                                    spliced_downconnect_ += [_downconnect]
                                    # higher row of Ps that form add_derPs:
                                    _P_ = [P for P_ in (__PP.P__[-max_rng:] + PP.P__[:max_rng]) for P in P_]
                                    # multiple lower-row Ps, [:] to not reference PP.P__, to be packed in accum_PP:
                                    P__ = [__P_[:] for __P_ in __PP.P__[-max_rng-1:]] + [P_[:] for P_ in PP.P__[:max_rng]]
                                    # comp Ps x gap _PP:
                                    for P_ in P__:  # lower row
                                        for _P in _P_[:]:  # higher row
                                            for P in P_[:]:  # lower Ps per _P
                                                if isinstance(P, CPP): add_derP = comp_layer(_P, P)
                                                else:                  add_derP = comp_P(_P, P)
                                                accum_PP(__PP, add_derP)
                                                # we may be adding multiple PPs here, one per x-overlap?
                                        _P_ = P_

                                    # merge PP into __PP
                                    for derP_ in PP.derP__:
                                        for derP in derP_:
                                            _derP__ = [_pri_derP for _pri_derP_ in __PP.derP__ for _pri_derP in _pri_derP_]  # accum_PP may append new derP
                                            if derP not in _derP__:
                                                accum_PP(__PP, derP)  # accumulate params

                                    # update upconnect and downconnect reference for _PP, pack into function?
                                    for up_PP in _PP.upconnect_:
                                        if up_PP not in __PP.upconnect_:  # PP may have multiple downconnects
                                            __PP.upconnect_.append(up_PP)
                                            Rdn = 0
                                            n = 0
                                            for up_PP in _PP.upconnect_:
                                                if up_PP not in __PP.upconnect_:  # PP may have multiple downconnects
                                                    __PP.upconnect_.append(up_PP)
                                                    Rdn += up_PP.dP > __PP.mP
                                                    n += 1
                                            __PP.rdn += Rdn / n
                                            __PP.rdn += up_PP.dP > __PP.mP  # ?

                                    for i, down_PP in enumerate(_PP.downconnect_):
                                        if _PP in down_PP.upconnect_:
                                            down_PP.upconnect_[down_PP.upconnect_.index(_PP)] = __PP  # update lower PP's upconnect from PP to _PP
                                            if down_PP not in _PP.downconnect_:
                                                __PP.downconnect_ += [down_PP]
                                                __PP.rdn += down_PP.dP > __PP.mP

                                    # update upconnect and downconnect reference for PP
                                    for up_PP in PP.upconnect_:
                                        if up_PP not in __PP.upconnect_:  # PP may have multiple downconnects
                                            __PP.upconnect_.append(up_PP)
                                            __PP.rdn += up_PP.dP > __PP.mP
                                    for i, down_PP in enumerate(PP.downconnect_):
                                        if PP in down_PP.upconnect_:
                                            down_PP.upconnect_[down_PP.upconnect_.index(PP)] = __PP  # update lower PP's upconnect from PP to _PP
                                            if down_PP not in _PP.downconnect_:
                                                __PP.downconnect_ += [down_PP]
                                                __PP.rdn += down_PP.dP > __PP.mP

                                    fbreak = 1
                                    break  # break PP_ loop
                        if fbreak: break   # break  _PP loop

                    for downconnect in spliced_downconnect_:  # PP__.downconnect could be PP or derP
                        __PP.downconnect_.remove(downconnect)

        else:  # der+ fork
            pass


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


def comp_layer(_derP, derP):

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

    return CderP(x0=x0, L=L, y=_derP.y, mP=mP, dP=dP, params=derivatives, P=derP, _P=_derP)

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