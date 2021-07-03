'''
  line_patterns is a principal version of 1st-level 1D algorithm

  Operations:
- Cross-compare consecutive pixels within each row of image, forming dert_: queue of derts, each a tuple of derivatives per pixel.
  dert_ is then segmented into patterns Pms and Pds: contiguous sequences of pixels forming same-sign match or difference.
  Initial match is inverse deviation of variation: m = ave_|d| - |d|, rather than minimum for directly defined match:
  albedo or intensity of reflected light doesn't correlate with predictive value of the object that reflects it.
  -
- Match patterns Pms are spans of inputs forming same-sign match. Positive Pms contain high-match pixels, which are likely
  to match more distant pixels. Thus, positive Pms are evaluated for cross-comp of pixels over incremented range.
  -
- Difference patterns Pds are spans of inputs forming same-sign ds. d sign match is a precondition for d match, so only
  same-sign spans (Pds) are evaluated for cross-comp of constituent differences, which forms higher derivatives.
  (d match = min: rng+ comp value: predictive value of difference is proportional to its magnitude, although inversely so)
  -
  Both extended cross-comp forks are recursive: resulting sub-patterns are evaluated for deeper cross-comp, same as top patterns.
  These forks here are exclusive per P to avoid redundancy, but they overlap in line_patterns_olp.
'''

# add ColAlg folder to system path
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname("CogAlg"), '..')))

import cv2
import argparse
from time import time
from utils import *
from itertools import zip_longest
from frame_2D_alg.class_cluster import ClusterStructure, NoneType, comp_param, Cdm

class Cdert(ClusterStructure):
    p = int
    d = int
    m = int

class CP(ClusterStructure):

    sign = bool
    L = int
    I = int
    D = int
    M = int
    x0 = int
    p_ = list
    d_ = list
    m_ = list
    ip_ = list  # only used for range_comp?
    sublayers = list
    # for line_PPs
    derP = object  # forward comp_P derivatives
    _smP = bool  # backward mP sign, for derP.sign determination, not needed thereafter
    fPd = bool  # P is Pd if true, else Pm

# pattern filters or hyper-parameters: eventually from higher-level feedback, initialized here as constants:

ave = 15  # |difference| between pixels that coincides with average value of Pm
ave_min = 2  # for m defined as min |d|: smaller?
ave_M = 50  # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 5  # min |D| for initial incremental-derivation comparison(d_)
ave_nP = 5  # average number of sub_Ps in P, to estimate intra-costs? ave_rdn_inc = 1 + 1 / ave_nP # 1.2
ave_rdm = .5  # average dm / m, to project bi_m = m * 1.5
init_y = 0  # starting row, the whole frame doesn't need to be processed

'''
    Conventions:
    postfix '_' denotes array name, vs. same-name elements
    prefix '_' denotes prior of two same-name variables
    prefix 'f' denotes flag
    capitalized variables are normally summed small-case variables
'''

def cross_comp(frame_of_pixels_):  # converts frame_of_pixels to frame_of_patterns, each pattern maybe nested

    Y, X = frame_of_pixels_.shape  # Y: frame height, X: frame width
    frame_of_patterns_ = []

    for y in range(init_y + 1, Y):  # y is index of new line pixel_, a brake point here, we only need one row to process
        d_ = []  # initialization
        pixel_ = frame_of_pixels_[y, :]
        _p = pixel_[0]

        for p in pixel_[1:]:  # pixel p is compared to prior pixel _p in a row
            d_.append(p -_p)  # m = ave - abs(d) is defined for accumulation only, otherwise redundant
            _p = p

        Pm_ = form_P_(pixel_, pixel_, d_, fPd=False)  # forms m-sign patterns
        if len(Pm_) > 4:
            adj_M_ = form_adjacent_M_(Pm_)  # compute adjacent Ms to evaluate contrastive borrow potential
            intra_Pm_(Pm_, adj_M_, fid=False, rdn=1, rng=2)  # rng is unilateral, evaluates for sub-recursion per Pm

        frame_of_patterns_.append(Pm_)
        # line of patterns is added to frame of patterns

    return frame_of_patterns_  # frame of patterns will be output to level 2


def form_P_(ip_, p_, d_, fPd):  # initialization, accumulation, termination

    P_ = []  # initialization:
    x=0
    _d = d_[0]
    _p = p_[0]
    _m = ave - abs(_d)
    if fPd: _sign = _d > 0
    else: _sign = _m > 0

    P = CP(sign=_sign, L=1, I=_p, D=_d, M=_m, x0=0, p_=[_p], d_=[_d], m_=[_m], ip=-p_[0], sublayers=[], _smP=False, fPd=fPd)
    # segment by m sign:
    for p, d in zip(p_[1:], d_[1:] ):
        m = ave - abs(d)
        if fPd: sign = d > 0
        else: sign = m > 0

        if sign != _sign:  # sign change, terminate P
            P_.append(P)
            # re-initialization:
            P = CP(sign=_sign, L=1, I=p, D=d, M=m, x0=x-(P.L-1), p_=[p], d_=[d], m_=[m], ip=ip_[x], sublayers=[], _smP=False, fPd=fPd)
        else:
            # accumulate params:
            P.L += 1; P.I += p; P.D += d; P.M += m
            P.p_ += [p]; P.d_ += [d]
            P.ip_ += [ip_[x]]  # only used for range_comp?

        _sign = sign
        x += 1

    P_.append(P)  # last incomplete P
    return P_


def form_adjacent_M_(Pm_):  # compute array of adjacent Ms, for contrastive borrow evaluation
    '''
    Value is projected match, while variation has contrast value only: it matters to the extent that it interrupts adjacent match: adj_M.
    In noise, there is a lot of variation. but no adjacent match to cancel, so variation in noise has no predictive value.
    On the other hand, we may have a 2D outline or 1D contrast with low gradient / difference, but it terminates adjacent uniform span.
    That contrast may be salient if it can borrow sufficient predictive value from that adjacent high-match span.
    '''

    pri_M = Pm_[0].M  # comp_g value is borrowed from adjacent opposite-sign Ms
    M = Pm_[1].M
    adj_M_ = [abs(Pm_[1].M)]  # initial next_M, no / 2: projection for first P, abs for bilateral adjustment

    for Pm in Pm_[2:]:
        next_M = Pm.M
        adj_M_.append((abs(pri_M / 2) + abs(next_M / 2)))  # exclude M
        pri_M = M
        M = next_M
    adj_M_.append(abs(pri_M))  # no / 2: projection for last P

    return adj_M_

''' 
    Recursion in intra_P extends pattern with sub_: hierarchy of sub-patterns, to be adjusted by macro-feedback:
    # P:
    sign,  # of m | d 
    dert_, # buffer of elements, input for extended cross-comp
    # next fork:
    fPd, # flag: select Pd vs. Pm forks in form_P_
    fid, # flag: input is derived: magnitude correlates with predictive value: m = min-ave, else m = ave-|d|
    rdn, # redundancy to higher layers, possibly lateral overlap of rng+ & der+, rdn += 1 * typ coef?
    rng, # comp range
    sublayers: # multiple layers of sub_P_s from d segmentation or extended comp, nested to depth = sub_[n]
               # for layer-parallel access and comp, as in frequency domain representation
               # orders of composition: 1st: dert_, 2nd: sub_P_[ derts], 3rd: sublayers[ sub_Ps[ derts]] 
'''

def intra_Pm_(P_, adj_M_, fid, rdn, rng):  # evaluate for sub-recursion in line Pm_, pack results into sub_Pm_

    comb_layers = []  # combine into root P sublayers[1:]

    for P, adj_M in zip(P_, adj_M_):  # each sub_layer is nested to depth = sublayers[n]
        if P.L > 2 ** rng:
            if P.sign:  # +Pm: low-variation span, eval comp at rng=2^n: 1, 2, 3; kernel size 2, 4, 8...
                if P.M - adj_M > ave_M * rdn:  # reduced by lending to contrast: all comps form params for hLe comp?
                    '''
                    if localized filters:
                    P_ave = (P.M - adj_M) / P.L  
                    loc_ave = (ave - P_ave) / 2  # ave is reduced because it's for inverse deviation, possibly negative?
                    loc_ave_min = (ave_min + P_ave) / 2
                    rdert_ = range_comp(P.dert_, loc_ave, loc_ave_min, fid)
                    '''
                    p_, rp_, rd_ = range_comp(P.p_, P.p_, P.d_)  # rng+ comp with localized ave, skip predictable next dert
                    sub_Pm_ = form_P_(p_, rp_, rd_, fPd=False)  # cluster by m sign
                    Ls = len(sub_Pm_)
                    P.sublayers += [[(Ls, False, fid, rdn, rng, sub_Pm_, [], [])]]  # sub_PPm_, sub_PPd_
                    # 1st layer, Dert=[], fill if Ls > min?
                    if len(sub_Pm_) > 4:
                        sub_adj_M_ = form_adjacent_M_(sub_Pm_)
                        P.sublayers += intra_Pm_(sub_Pm_, sub_adj_M_, fid, rdn+1 + 1/Ls, rng+1)  # feedback
                        # add param summation within sublayer, for comp_sublayers?
                        # splice sublayers across sub_Ps:
                        comb_layers = [comb_layers + sublayers for comb_layers, sublayers in
                                       zip_longest(comb_layers, P.sublayers, fillvalue=[])]

            else:  # -Pm: high-variation span, min neg M is contrast value, borrowed from adjacent +Pms:
                if min(-P.M, adj_M) > ave_D * rdn:  # cancelled M+ val, M = min | ~v_SAD

                    rel_adj_M = adj_M / -P.M  # for allocation of -Pm' adj_M to each of its internal Pds
                    sub_Pd_ = form_P_(P.p_, P.p_, P.d_, fPd=True)  # cluster by input d sign match: partial d match
                    Ls = len(sub_Pd_)
                    P.sublayers += [[(Ls, True, True, rdn, rng, sub_Pd_)]]  # 1st layer, Dert=[], fill if Ls > min?

                    P.sublayers += intra_Pd_(sub_Pd_, rel_adj_M, rdn+1 + 1/Ls, rng)  # der_comp eval per nPm
                    # splice sublayers across sub_Ps, for return as root sublayers[1:]:
                    comb_layers = [comb_layers + sublayers for comb_layers, sublayers in
                                   zip_longest(comb_layers, P.sublayers, fillvalue=[])]

    return comb_layers


def intra_Pd_(Pd_, rel_adj_M, rdn, rng):  # evaluate for sub-recursion in line P_, packing results in sub_P_

    comb_layers = []
    for P in Pd_:  # each sub in sub_ is nested to depth = sub_[n]

        if min(abs(P.D), abs(P.D) * rel_adj_M) > ave_D * rdn and P.L > 3:  # abs(D) * rel_adj_M: allocated adj_M
            # cross-comp of ds:
            dd_ = deriv_comp(P.d_)
            sub_Pm_ = form_P_(P.p_, P.d_, dd_, fPd=True)  # cluster Pd derts by md, won't happen
            Ls = len(sub_Pm_)
            # 1st layer: Ls, fPd, fid, rdn, rng, sub_P_, sub_PPm_, sub_PPd_:
            P.sublayers += [[(Ls, True, True, rdn, rng, sub_Pm_, [], [] )]]

            if len(sub_Pm_) > 3:
                sub_adj_M_ = form_adjacent_M_(sub_Pm_)
                P.sublayers += intra_Pm_(sub_Pm_, sub_adj_M_, 1, rdn+1 + 1/Ls, rng + 1)
                # splice sublayers across sub_Ps:
                comb_layers = [comb_layers + sublayers for comb_layers, sublayers in
                               zip_longest(comb_layers, P.sublayers, fillvalue=[])]
    ''' 
    adj_M is not affected by primary range_comp per Pm?
    no comb_m = comb_M / comb_S, if fid: comb_m -= comb_|D| / comb_S: alt rep cost
    same-sign comp: parallel edges, cross-sign comp: M - (~M/2 * rL) -> contrast as 1D difference?
    '''
    return comb_layers


def range_comp(p_, rp_, rd_):
    # no rp_, rd_ = [], []: should be cumulative? unless preserve rim layers 

    rng_p_ = []
    rng_d_ = []

    p_ = p_[::2]  # sparse p_ and d_, skipping odd ps compared in prior rng: 1 skip / 1 add, to maintain 2x overlap
    rp_ = rp_[::2]
    rd_ = rd_[::2]
    _p = p_[0] # get 1st element
    _pri_rng_p = rp_[0]
    _pri_rng_d = rd_[0]
    rng_p_.append(_pri_rng_p) # pack 1st rng p and d
    rng_d_.append(_pri_rng_d)

    for p, pri_rng_p, pri_rng_d in zip(p_[1:], rp_[1:], rd_[1:]):
        d = p -_p
        rng_p = _p + pri_rng_p  # intensity accumulated in rng
        rng_d = d + pri_rng_d  # difference accumulated in rng
        rng_p_.append(rng_p)
        rng_d_.append(rng_d)  
        _p = p
        
    return p_, rng_p_, rng_d_



def deriv_comp(d_):  # cross-comp consecutive uni_ds in same-sign dert_: sign match is partial d match
    # dd and md may match across d sign, but likely in high-match area, spliced by spec in comp_P?

    dd_ = []  # initialization:
    __i = abs(d_[0])  # each prefix '_' denotes prior
    _i = abs(d_[1])

    _d = _i - __i  # initial comp

    dd_.append(0)
    
    for d in d_[2:]:
        i = abs(d)  # unilateral d, same sign in Pd
        d = i - _i  # d is dd
        dd_.append(_d) # unilateral _d 
        _i, _d = i, d 

    dd_.append(_d)

    return dd_

'''
# not revised:
def deriv_comp(dert_):  # cross-comp consecutive uni_ds in same-sign dert_: sign match is partial d match
    # dd and md may match across d sign, but likely in high-match area, spliced by spec in comp_P?

    ddert_ = []  # initialization:
    __i = dert_[1].d  # each prefix '_' denotes prior
    _i = dert_[2].d

    __i = abs(__i);  _i = abs(_i)
    _d = _i - __i  # initial comp
    _m = min(__i, _i) - ave_min
    ddert_.append(Cdert(p=_i, d=0, m=(_m + _m / 2)))  # no __d, back-project __m = _m * .5

    for dert in dert_[3:]:
        i = abs(dert.d)  # unilateral d, same sign in Pd
        d = i - _i  # d is dd
        m = min(i, _i) - ave_min  # md = min: magnitude of derived vars corresponds to predictive value
        ddert_.append(Cdert(p=_i, d=_d, m=_m + m))  # unilateral _d and bilateral m per _i
        _i, _d, _m = i, d, m

    ddert_.append(Cdert(p=_i, d=_d, m=(_m + _m / 2)))  # forward-project bilateral m
    return ddert_
'''

def cross_comp_spliced(frame_of_pixels_):  # converts frame_of_pixels to frame_of_patterns, each pattern maybe nested
    '''
    process all image rows as a single line, vertically consecutive and preserving horizontal direction
    '''
    Y, X = frame_of_pixels_.shape  # Y: frame height, X: frame width
    pixel__ = []

    for y in range(init_y + 1, Y):  # y is index of new line
        pixel__.append([ frame_of_pixels_[y, :] ])  # splice all rows into pixel__

    # initialization:
    dert_ = []
    __p, _p = pixel__[0:2]  # each prefix '_' denotes prior
    _d = _p - __p  # initial comparison
    _m = ave - abs(_d)
    dert_.append( Cdert(p=__p, d=0, m=(_m + _m / 2)))  # project _m to bilateral m, first dert is for comp_P only?

    for p in pixel__[2:]:  # pixel p is compared to prior pixel _p in a row
        d = p - _p
        m = ave - abs(d)  # initial match is inverse deviation of |difference|
        dert_.append( Cdert(p=_p, d=_d, m=m + _m))  # pack dert: prior p, prior d, bilateral match
        _p, _d, _m = p, d, m
    dert_.append( Cdert(p=_p, d=_d, m=(_m + _m / 2)))  # unilateral d, forward-project last m to bilateral m

    Pm_ = form_Pm_(dert_)  # forms m-sign patterns
    if len(Pm_) > 4:
        adj_M_ = form_adjacent_M_(Pm_)  # compute adjacent Ms to evaluate contrastive borrow potential
        intra_Pm_(Pm_, adj_M_, fid=False, rdn=1, rng=3)  # evaluates for sub-recursion per Pm

    return Pm_  # frame of patterns, an output to line_PPs (level 2 processing)


if __name__ == "__main__":
    # Parse argument (image)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image',
                                 help='path to image file',
                                 default='.//raccoon.jpg')
    arguments = vars(argument_parser.parse_args())
    # Read image
    image = cv2.imread(arguments['image'], 0).astype(int)  # load pix-mapped image
    assert image is not None, "No image in the path"
    image = image.astype(int)

    start_time = time()
    # Main
    frame_of_patterns_ = cross_comp(image)  # returns Pm__

    fline_PPs = 1
    if fline_PPs:  # debug line_PPs_draft
        from line_PPs2_draft import *
        frame_PP_ = []

        for y, P_ in enumerate(frame_of_patterns_):
            PPm_, PPd_ = search(P_)
            frame_PP_.append([PPm_, PPd_])

    end_time = time() - start_time
    print(end_time)