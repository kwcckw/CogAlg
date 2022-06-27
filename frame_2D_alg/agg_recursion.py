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

# agg-recursive versions should be more complex?
class CderPP(ClusterStructure):  # tuple of derivatives in PP uplink_ or downlink_, PP can also be PPP, etc.

    # draft
    params = list  # PP derivation layer, flat, decoded by mapping each m,d to lower-layer param
    x0 = int  # redundant to params:
    x = float  # median x
    L = int  # pack in params?
    sign = NoneType  # g-ave + ave-ga sign
    y = int  # for vertical gaps in PP.P__, replace with derP.P.y?
    PP = object  # lower comparand
    _PP = object  # higher comparand
    root = lambda:None  # segment in sub_recursion
    # higher derivatives
    rdn = int  # mrdn, + uprdn if branch overlap?
    uplink_layers = lambda: [[],[]]  # init a layer of dderPs and a layer of match_dderPs
    downlink_layers = lambda: [[],[]]
   # from comp_dx
    fdx = NoneType

class CPPP(CPP, CderPP):

    # draft
    params = list  # derivation layers += derP params per der+, param L is actually Area
    sign = bool
    rng = lambda: 1  # rng starts with 1
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs
    Rdn = int  # for accumulation only
    nP = int  # len 2D derP__ in levels[0][fPd]?  ly = len(derP__), also x, y?
    uplink_layers = lambda: [[],[]]
    downlink_layers = lambda: [[],[]]
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    box = list  # for visualization only, original box before flipping
    mask__ = bool
    P__ = list  # input  # derP__ = list  # redundant to P__
    seg_levels = lambda: [[[]],[[]]]  # from 1st agg_recursion, seg_levels[0] is seg_t, higher seg_levels are segP_t s
    PPP_levels = list  # from 2nd agg_recursion, PP_t = levels[0], from form_PP, before recursion
    layers = list  # from sub_recursion, each is derP_t
    root = lambda:None  # higher-order segP or PPP


def agg_recursion(blob, fseg):  # compositional recursion per blob.Plevel. P, PP, PPP are relative terms, each may be of any composition order

    if fseg: PP_t = [blob.seg_levels[0][-1], blob.seg_levels[1][-1]]   # blob is actually PP, recursion forms segP_t, seg_PP_t, etc.
    else:    PP_t = [blob.levels[0][-1], blob.levels[1][-1]]  # input-level composition Ps, initially PPs

    PPP_t = []  # next-level composition Ps, initially PPPs  # for fiPd, PP_ in enumerate(PP_t): fiPd = fiPd % 2  # dir_blob.M += PP.M += derP.m
    n_extended = 0

    for i, PP_ in enumerate(PP_t):   # fiPd = fiPd % 2
        fiPd = i % 2
        if fiPd: ave_PP = ave_dPP
        else:    ave_PP = ave_mPP
        if fseg: M = ave- blob.params[-1][fiPd][4]  # blob.params[0][fiPd][4] is mG | dG
        else: M = ave-abs(blob.G)  # if M > ave_PP * blob.rdn and len(PP_)>1:  # >=2 comparands

        if len(PP_)>1:
            n_extended += 1
            derPP_t = comp_PP_(PP_)  # compare all PPs to the average (centroid) of all other PPs, is generic for lower level
            PPP_t = form_PPP_t(derPP_t)
            # call individual comp_PP if mPPP > ave_mPPP, converting derPP to CPPP
            splice_PPs(PPP_t)  # for initial PPs only: if PP is CPP?
            sub_recursion_eval(PPP_t)  # rng+ or der+, if PP is CPPP?
        else:
            PPP_t += [[], []]  # replace with neg PPPs?

    if fseg: blob.seg_levels += [PPP_t]  # new level of segPs
    else:    blob.levels += [PPP_t]  # levels of dir_blob are Plevels

    if n_extended/len(PP_t) > 0.5:  # mean ratio of extended PPs
        agg_recursion(blob, fseg)
'''
- Compare each PP to the average (centroid) of all other PPs in PP_, or maximal cartesian distance, forming derPPs.  
- Select above-average derPPs as PPPs, representing summed derivatives over comp range, overlapping between PPPs.
'''

def comp_PP_(PP_):  # PP can also be PPP, etc.
    derPPm_, derPPd_ = [],[]

    for PP in PP_:
        compared_PP_ = copy(PP_)  # shallow copy
        compared_PP_.remove(PP)
        n = len(compared_PP_)

        summed_params = deepcopy(compared_PP_[0].params)  # sum same-type params across compared PPs, init with 1st element
        for compared_PP in compared_PP_[1:]:
            # generic unpack and process, summed_params accum over compared_PP_:
            summed_params += [func_layers(summed_params, compared_PP.params, out_layers=summed_params, func=accum_ptuple)]
        ave_params = [ave_layers(summed_params, n, [])]

        derPP = CPP(params=deepcopy(PP.params), layers=[PP_])  # derPP inherits PP.params
        der_layers = []
        derPP.params += [func_layers(PP.params, ave_params, out_layers=der_layers, func=comp_ptuple)]  # generic unpack,function
        '''
        comp to ave params of compared PPs, form new layer: derivatives of all lower layers, 
        initial 3 layer nesting diagram: https://github.com/assets/52521979/ea6d436a-6c5e-429f-a152-ec89e715ebd6
        '''
        derPPm_.append(copy_P(derPP, Ptype=2))
        derPPd_.append(copy_P(derPP, Ptype=2))

    return derPPm_, derPPd_


def form_PPP_t(derPP_t):  # form PPs from match-connected segs
    PPP_t = []

    for fPd, derPP_ in enumerate(derPP_t):
        # sort by value of last layer: derivatives of all lower layers:
        derPP_ = sorted(derPP_, key=lambda derPP: derPP.params[-1][fPd], reverse=True)  # descending order
        PPP_ = []
        for i, derPP in enumerate(derPP_):
            derPP_val = 0
            for param_layer in derPP.params:  # may need recursive unpack here
                derPP.rdn += param_layer[fPd] > param_layer[1-fPd]
                derPP_val += param_layer[fPd]  # make it a param?

            ave = vaves[fPd] * derPP.rdn * (i+1)  # derPP is redundant to higher-value previous derPPs in derPP_
            if derPP_val > ave:
                PPP_ += [derPP]  # base derPP and PPP is CPP
                if derPP_val > ave*10:
                    ind_comp_PP_(derPP, fPd)  # derPP is converted from CPP to CPPP
            else:
                break  # ignore below-ave PPs
        PPP_t.append(PPP_)
    return PPP_t

# draft
def ind_comp_PP_(_PP, fPd):  # 1-to-1 comp, _PP is converted from CPP to higher-composition CPPP

    derPP_ = []
    rng = _PP.params[-1][fPd] / 3  # 3: ave per rel_rng+=1, actual rng is Euclidean distance:

    for PP in _PP.layers[0]:  # 1-to-1 comparison between _PP and other PPs within rng
        derPP = CderPP()
        _area = _PP.params.L  # pseudo, we need L index in params
        area = PP.params.L
        dx = _PP.x/_area - PP.x/area
        dy = _PP.y/_area - PP.y/area
        distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids
        _val = _PP.params[-1][fPd]
        val = PP.params[-1][fPd]

        if distance / ((_val+val)/2) < rng:  # distance relative to value, vs. area?

            der_layers = [func_layers(_PP.params, PP.params, out_layers=[], func=comp_ptuple)]  # each layer is sub_layers
            _PP.downlink_layers += [der_layers]
            PP.uplink_layers += [der_layers]
            derPP_ += [derPP]

    for i, _derPP in enumerate(derPP_):  # cluster derPPs into PPPs by connectivity, overwrite derPP[i]

        if _derPP.params[-1][fPd]:
            PPP = CPPP(params=deepcopy(_derPP.params), layers=[_derPP.PP])
            PPP.accum_from(_derPP)  # initialization
            _derPP.root = PPP
            for derPP in derPP_[i+1:]:
                if not derPP.PP.root:
                    if derPP.params[-1][fPd]:  # positive and not in PPP yet
                        PPP.layers.append(derPP)  # multiple composition orders
                        PPP.accum_from(_derPP)
                        derPP.root = PPP
                    # pseudo:
                    elif sum([derPP.params[:-1]][fPd]) > ave*len(derPP.params)-1:
                         # splice PP and their segs
                         pass
    '''
    if derPP.match params[-1]: form PPP
    elif derPP.match params[:-1]: splice PPs and their segs? 
    '''

def func_layers(_layers, layers, out_layers, func):

    # recursive unpack of nested ptuple pairs, if any from der+, in the bottom layer or sublayer:
    out_layers += [func_pairs(_layers[0], layers[0], out_pairs=[], func_ptuple=func)]

    # recursive unpack of deeper layers, from agg+ in 3rd and higher layers, down to nested tuple pairs
    for _layer, layer in zip(_layers[1:], layers[1:]):
        out_layers += [func_layers(_layer, layer, out_layers, func)]  # layer = deeper sub_layers
    '''
    1st and 2nd layers are single sublayers, the 2nd adds tuple pair nesting. Both are unpacked by func_pairs, not func_layers.  
    Multiple sublayers start on the 3rd layer, because it's derived from comparison between two (not one) lower layers. 
    4th layer is derived from comparison between 3 lower layers, where the 3rd layer is already nested, etc.
    '''
    return out_layers # possibly nested param layers


def func_pairs(_pairs, pairs, out_pairs, func_ptuple):  # recursively unpack m,d tuple pairs from der+

    if isinstance(_pairs[0], list):  # pairs is a pair, possibly nested
        out_pairs += func_pairs(_pairs[0], pairs[0], out_pairs, func_ptuple)
    else:
        out_pairs += func_ptuple(_pairs[0], pairs[0])  # pairs is actually a ptuple, 1st element is a param

    return out_pairs  # possibly nested m,d ptuple pairs


def ave_layers(summed_params, n, ave_params):

    # recursive unpack of nested ptuple pairs, if any from der+:
    ave_params += [ave_pairs(summed_params, n, ave_pairs=[])]

    for summed_layer in summed_params[1:]:  # recursive unpack of deeper layers, if any from agg+:
        ave_params += [ave_layers(summed_layer, n, ave_params)]  # each layer is deeper sub_layers

def ave_pairs(pairs, n, ave_pairs):  # recursively unpack m,d tuple pairs from der+

    if isinstance(pairs[0], list):  # pairs is a pair, possibly nested
        ave_pairs += ave_pairs(pairs[0], n, ave_pairs)
    else:
        for i, param in enumerate(ave_pairs):  # pairs is actually a ptuple, 1st element is a param
            ave_pairs[i] = param / n  # 1st layer is latuple, decoded in func

    return ave_pairs  # possibly nested m,d ptuple pairs

'''
to be updated:
'''
# unpack and and accum same-type params
# use the same unpack sequence as in comp_layer?
def sum_nested_layer(sum_layer, params_layer):

    if isinstance(sum_layer[0], list):  # if nested, continue to loop and search for deeper list
        for j, (sub_sum_layer, sub_params_layer) in enumerate(zip(sum_layer, params_layer)):
            sum_nested_layer(sub_sum_layer, sub_params_layer)
    else:  # if layer is not nested, sum params
        for j, param in enumerate(params_layer):
            sum_layer[j] += param

# get average value for each param according to n value
def get_layers_average(sum_params, n):

    average_params = deepcopy(sum_params)  # get a copy as output
    if isinstance(average_params[0], list):  # if nested, continue to loop and search for deeper list
        for j, sub_sum_layer in enumerate(average_params):
            get_layers_average(sub_sum_layer, n)
    else:  # if layer is not nested, get average of each value
        for j, param in enumerate(average_params):
            average_params[j] = param/n

    return average_params


def comp_nested_layer(_param_layer, param_layer):

    if isinstance(_param_layer[0], list):   # if nested, continue to loop and search for deeper list
        sub_ders = []
        for j, (_sub_layer, sub_layer) in enumerate( zip(_param_layer, param_layer)):
            sub_ders += [comp_nested_layer(_sub_layer, sub_layer)]
        return sub_ders
    else:  # comp params if layer is not nested
        params, _, _ = comp_params(_param_layer, param_layer, nparams=len(_param_layer))
        mparams = params[0::2]  # get even index m params
        dparams = params[1::2]  # get odd index d params
        return [mparams, dparams]


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

