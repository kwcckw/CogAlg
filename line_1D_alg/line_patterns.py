'''
  line_patterns is a principal version of 1st-level 1D algorithm
  Operations:
  -
- Cross-compare consecutive pixels within each row of image, forming dert_: queue of derts, each a tuple of derivatives per pixel.
  dert_ is then segmented into patterns Pms and Pds: contiguous sequences of pixels forming same-sign match or difference.
  Initial match is inverse deviation of variation: m = ave_|d| - |d|,
  rather than a minimum for directly defined match: albedo of an object doesn't correlate with its predictive value.
  -
- Match patterns Pms are spans of inputs forming same-sign match. Positive Pms contain high-match pixels, which are likely
  to match more distant pixels. Thus, positive Pms are evaluated for cross-comp of pixels over incremented range.
  -
- Difference patterns Pds are spans of inputs forming same-sign ds. d sign match is a precondition for d match, so only
  same-sign spans (Pds) are evaluated for cross-comp of constituent differences, which forms higher derivatives.
  (d match = min: rng+ comp value: predictive value of difference is proportional to its magnitude, although inversely so)
  -
  Both extended cross-comp forks are recursive: resulting sub-patterns are evaluated for deeper cross-comp, same as top patterns.
  These forks here are exclusive per P to avoid redundancy, but they do overlap in line_patterns_olp.
'''

# add ColAlg folder to system path
import sys
from os.path import dirname, join, abspath

from numpy import int16, int32
sys.path.insert(0, abspath(join(dirname("CogAlg"), '..')))
import cv2
# import argparse
import pickle
from time import time
from matplotlib import pyplot as plt
from itertools import zip_longest
from frame_2D_alg.class_cluster import ClusterStructure, NoneType, comp_param
from line_PPs import *

class Cdert(ClusterStructure):
    i = int  # input for range_comp only
    p = int  # accumulated in rng
    d = int  # accumulated in rng
    m = int  # distinct in deriv_comp only

class CP(ClusterStructure):
    L = int
    I = int
    D = int
    M = int
    x0 = int
    dert_ = list  # contains (i, p, d, m)
    # for layer-parallel access and comp, ~ frequency domain, composition: 1st: dert_, 2nd: sub_P_[ dert_], 3rd: sublayers[ sub_P_[ dert_]]:
    sublayers = list  # multiple layers of sub_P_s from d segmentation or extended comp, nested to depth = sub_[n]
    subDerts = list  # conditionally summed [L,I,D,M] per sublayer, most are empty
    derDerts = list  # for compared subDerts only
    fPd = bool  # P is Pd if true, else Pm; also defined per layer
    # for rval Pp
    rval_ = list

verbose = False
# pattern filters or hyper-parameters: eventually from higher-level feedback, initialized here as constants:
ave = 15  # |difference| between pixels that coincides with average value of Pm
ave_min = 2  # for m defined as min |d|: smaller?
ave_M = 20  # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 5  # min |D| for initial incremental-derivation comparison(d_)
ave_nP = 5  # average number of sub_Ps in P, to estimate intra-costs? ave_rdn_inc = 1 + 1 / ave_nP # 1.2
ave_rdm = .5  # obsolete: average dm / m, to project bi_m = m * 1.5
ave_splice = 50  # to merge a kernel of 3 adjacent Ps
init_y = 0  # starting row, the whole frame doesn't need to be processed
halt_y = 999999999  # ending row
'''
    Conventions:
    postfix '_' denotes array name, vs. same-name elements
    prefix '_' denotes prior of two same-name variables
    prefix 'f' denotes flag
    capitalized variables are normally summed small-case variables
'''



def cross_comp_(input__, feedback_H, level, fPd, frecursive):
    '''
    cross comp input recursively
    '''

    output__ = []    # current level output
    Y = len(input__) # Y: frame height

    for y in range(init_y, min(halt_y, Y)):  # y is index of new row pixel_, we only need one row, use init_y=0, halt_y=Y for full frame
        X = len(input__[y])  # X: frame width          
        if X>1: # non empty row's elements (pixels, Ps or Pps)
            element_ = input__[y]                              # get current line input
            compute_feedback(element_, feedback_H, level, fPd) # compute hierarchy feedback       
            dert_ = search_(element_, level, fPd)              # search & compare input elements, then compute derivatives
            pattern_ = form_pattern_(dert_, level, fPd)        # form patterns with same sign derivatives
            output__.append(pattern_)
            
    # max level = 5 now, due to haven't add any deeper level computation evaluation
    if frecursive and level<5: # next level computation
        cross_comp_(output__, feedback_H, level+1, fPd, frecursive) # current level output is next level input

# main search function
def search_(element_, level, fPd):
    
    if level == 0 : # line_Ps
        dert_ = search_pixel_(element_)
                
    elif level == 1: # line_PPs
        dert_ = search_pattern_(element_, fPd)  
    else: # > level 1        
        dert_ = []
        search_Pp_(element_, dert_, fPd, level)
                             
    return dert_

# level 0 search (or root level) - search pixels to get patterns
def search_pixel_(pixel_):
    dert_ = []  # line-wide i_, p_, d_, m__
    _i = pixel_[0]
    # pixel i is compared to prior pixel _i in a row:
    for i in pixel_[1:]:
        d = i - _i  # accum in rng
        p = i + _i  # accum in rng
        m = ave - abs(d)  # for consistency with deriv_comp output, else redundant
        dert_.append( Cdert( i=i, p=p, d=d, m=m) )
        _i = i
        
    return dert_
    
# level 1 search - search patterns to get Pp
def search_pattern_(P_, fPd):

    Ldert_, Idert_, Ddert_, Mdert_, dert1_, dert2_, LP_, IP_, DP_, MP_ = [], [], [], [], [], [], [], [], [], []
    param_derts_ = [Ldert_, Idert_, Ddert_, Mdert_]
    param_Ps_ = [LP_, IP_, DP_, MP_]

    for i, (_P, P, P2) in enumerate(zip(P_, P_[1:], P_[2:] + [None])):
        for param_dert_, param_P_, param_name, ave_param in zip(param_derts_, param_Ps_, param_names, aves):
            _param  = getattr(_P, param_name[0])
            param   = getattr(P , param_name[0])

            if param_name == "L_":  # div_comp for L because it's a higher order of scale:
                _L = _param; L= param
                rL = L / _L  # higher order of scale, not accumulated: no search, rL is directional
                int_rL = int( max(rL, 1/rL))
                frac_rL = max(rL, 1/rL) - int_rL
                mL = int_rL * min(L, _L) - (int_rL*frac_rL) / 2 - ave_mL  # div_comp match is additive compression: +=min, not directional
                Ldert_.append(Cdert( i=L, p=L + _L, d=rL, m=mL))
                param_P_.append(_P)

            elif (fPd and param_name == "D_") or (not fPd and param_name == "M_") : # step = 2
                if i < len(P_)-2: # P size is -2 and dert size is -1 when step = 2, so last 2 elements are not needed
                    param2 = getattr(P2, param_name[0])
                    param_dert_ += [comp_param(_param, param2, param_name, ave_param)]
                    dert2_ += [param_dert_[-1].copy()]
                    param_P_.append(_P)
                dert1_ += [comp_param(_param, param, param_name, ave_param)]

            elif not (not fPd and param_name == "I_") : # step = 1
                param_dert_ += [comp_param(_param, param, param_name, ave_param)]
                param_P_.append(_P)

    # comp x variable range, depending on M of Is
    if not fPd: Idert_, IP_ = search_param_(P_, ave_mI, rave=1)

    # Pdert__
    Pdert__ = [(Ldert_, LP_), (Idert_, IP_), (Ddert_, DP_), (Mdert_, MP_)]

    return Pdert__, dert1_, dert2_

# search for level 2 - search rval_Pp to get rval_Pp__ per param
def search_Pp_(rval_Pp__, iPp_, fPd, level):
    
    if level == 1:
        if len(rval_Pp__)>1: # at least 2 elements to search with step=2     
            return search_pattern_(rval_Pp__, fPd)
        else:
            return [] # preseve index
    else:  
        Pp_ = []
        for rval_Pp_ in rval_Pp__: 
            Pp_.append(search_Pp_(rval_Pp_, Pp_, fPd, level-1))
        iPp_.append(Pp_)

# main form pattern function
def form_pattern_(dert_, level, fPd):

    if level==0:  # input is dert  
         # form m-sign patterns, rootP=None:
        Pm_ = form_P_(None, dert_, rdn=1, rng=1, fPd=fPd)  # eval intra_Pm_ per Pm in
        pattern_ = Pm_
    
    elif level == 1: # input is pdert
        Pdert__, dert1_, dert2_,  = dert_ 
        rval_Pp__, Ppm__ = form_Pp_root(Pdert__, dert1_, dert2_, fPd)
        pattern_ = rval_Pp__

    else: # input is higher level pdert , > level 1
        pattern_ = []
        form_PPp_root(dert_, pattern_, level, fPd)
    
    return pattern_

# form pattern for level 2 and above
def form_PPp_root(dert_, ipattern_, level, fPd):
    if level == 1: # input is pdert
        if dert_:
            Pdert__, dert1_, dert2_  = dert_
            rval_Pp__, Ppm__ = form_Pp_root(Pdert__, dert1_, dert2_, fPd)
            return rval_Pp__
        else:
            return [] # preserve index

    # input is higher level pdert , > level 1
    elif dert_: # non empty dert
        pattern_ = []
        for pdert__ in dert_[0]: # dert_[0] due to added bracket during search function
            pattern_.append(form_PPp_root(pdert__, pattern_, level-1, fPd))
        ipattern_.append(pattern_)


def compute_feedback(element_, feedback_H, level, fPd):
    '''
    just a simple draft , far from complete
    '''
    
    fbM = fbL = 0
    '''
    for element in element_:
        fbM += element.M; fbL += element.L
        fbM += element.M; fbL += element.L
        if abs(fbM) > ave_Dave:
            if abs(fbM / fbL) > ave_dave:
                fbM = fbL = 0
                pass  # eventually feedback: line_patterns' cross_comp(frame_of_pixels_, ave + fbM / fbL)
                # also terminate Fspan: same-filter frame_ with summed params, re-init at all 0s

        element.I /= element.L; element.D /= element.L; element.M /= element.L  # immediate normalization to a mean
    '''
    return fbM, fbL

def update_aves(fbM, fbL):
    '''
    update aves here
    '''
    pass


def cross_comp(frame_of_pixels_):  # converts frame_of_pixels to frame_of_patterns, each pattern may be nested

    Y, X = frame_of_pixels_.shape  # Y: frame height, X: frame width
    frame_of_patterns_ = []
    '''
    if cross_comp_spliced:  # process all image rows as a single line, vertically consecutive and preserving horizontal direction:
        pixel_=[]; dert_=[]  
        for y in range(init_y, Y):  
            pixel_.append([ frame_of_pixels_[y, :]])  # splice all rows into pixel_
        _i = pixel_[0]
    else:
    '''
    for y in range(init_y, min(halt_y, Y)):  # y is index of new row pixel_, we only need one row, use init_y=0, halt_y=Y for full frame

        # initialization:
        dert_ = []  # line-wide i_, p_, d_, m__
        pixel_ = frame_of_pixels_[y, :]
        _i = pixel_[0]
        # pixel i is compared to prior pixel _i in a row:
        for i in pixel_[1:]:
            d = i - _i  # accum in rng
            p = i + _i  # accum in rng
            m = ave - abs(d)  # for consistency with deriv_comp output, else redundant
            dert_.append( Cdert( i=i, p=p, d=d, m=m) )
            _i = i
        # form m-sign patterns, rootP=None:
        Pm_ = form_P_(None, dert_, rdn=1, rng=1, fPd=False)  # eval intra_Pm_ per Pm in
        rval_Pm_ = form_rval_P_(Pm_)
        
        frame_of_patterns_.append(Pm_)  # add line of patterns to frame of patterns, skip if cross_comp_spliced

    return frame_of_patterns_  # frame of patterns is an input to level 2

def form_P_(rootP, dert_, rdn, rng, fPd):  # accumulation and termination, rdn and rng are pass-through intra_P_
    # initialization:
    P_ = []
    x = 0
    _sign = None  # to initialize 1st P, (None != True) and (None != False) are both True

    for dert in dert_:  # segment by sign
        if fPd: sign = dert.d > 0
        else:   sign = dert.m > 0
        if sign != _sign:
            # sign change, initialize and append P
            P = CP( L=1, I=dert.p, D=dert.d, M=dert.m, x0=x, dert_=[dert], sublayers=[], fPd=fPd)
            P_.append(P)  # updated with accumulation below
        else:
            # accumulate params:
            P.L += 1; P.I += dert.p; P.D += dert.d; P.M += dert.m
            P.dert_ += [dert]
        x += 1
        _sign = sign

    if rootP:  # call from intra_P_
        Dert = []
        # sublayers brackets: 1st: param set, 2nd: Dert, param set, 3rd: sublayer concatenated from n root_Ps, 4th: hierarchy
        rootP.sublayers = [[(fPd, rdn, rng, P_, [], [])]]  # 1st sublayer has one subset: sub_P_ param set, last[] is sub_Ppm__
        rootP.subDerts = [Dert]
        if len(P_) > 4:  # 2 * (rng+1) = 2*2 =4

            if rootP.M * len(P_) > ave_M * 5:  # or in line_PPs?
                Dert[:] = [0, 0, 0, 0]  # P. L, I, D, M summed within a layer
                for P in P_:  Dert[0] += P.L; Dert[1] += P.I; Dert[2] += P.D; Dert[3] += P.M

            comb_sublayers, comb_subDerts = intra_P_(P_, rdn, rng, fPd)  # deeper comb_layers feedback, subDerts is selective
            rootP.sublayers += comb_sublayers
            rootP.subDerts += comb_subDerts
    else:
        # call from cross_comp
        intra_P_(P_, rdn, rng, fPd)

    return P_  # used only if not rootP, else packed in rootP.sublayers and rootP.subDerts


def intra_P_(P_, rdn, rng, fPd):  # recursive cross-comp and form_P_ inside selected sub_Ps in P_

    adj_M_ = form_adjacent_M_(P_)  # compute adjacent Ms to evaluate contrastive borrow potential
    comb_sublayers = []
    comb_subDerts = []  # may not be needed, evaluation is more accurate in comp_sublayers?

    for P, adj_M in zip(P_, adj_M_):
        if P.L > 2 * (rng+1):  # vs. **? rng+1 because rng is initialized at 0, as all params
            rel_adj_M = adj_M / -P.M  # for allocation of -Pm' adj_M to each of its internal Pds?

            if fPd:  # P is Pd, -> sub_Pdm_, in high same-sign D span
                if min( abs(P.D), abs(P.D) * rel_adj_M) > ave_D * rdn:
                    ddert_ = deriv_comp(P.dert_)  # i is d
                    form_P_(P, ddert_, rdn+1, rng+1, fPd=True)  # cluster Pd derts by md sign, eval intra_Pm_(Pdm_), won't happen
            else:  # P is Pm,
                # +Pm -> sub_Pm_ in low-variation span, eval comp at rng=2^n: 1, 2, 3; kernel size 2, 4, 8..:
                if P.M > ave_M * rdn:  # no -adj_M: lend to contrast is not adj only, reflected in ave?
                    ''' if local ave:
                    loc_ave = (ave + (P.M - adj_M) / P.L) / 2  # mean ave + P_ave, possibly negative?
                    loc_ave_min = (ave_min + (P.M - adj_M) / P.L) / 2  # if P.M is min?
                    rdert_ = range_comp(P.dert_, loc_ave, loc_ave_min, fid)
                    '''
                    rdert_ = range_comp(P.dert_)  # rng+, skip predictable next dert, local ave? rdn to higher (or stronger?) layers
                    form_P_(P, rdert_, rdn+1, rng+1, fPd=False)  # cluster by m sign, eval intra_Pm_
                # -Pm -> sub_Pd_
                elif -P.M > ave_D * rdn:  # high-variation span, neg M is contrast, implicit borrow from adjacent +Pms, M=min
                    # or if min(-P.M, adj_M),  rel_adj_M = adj_M / -P.M  # allocate -Pm adj_M to each sub_Pd?
                    form_P_(P, P.dert_, rdn+1, rng, fPd=True)  # cluster by d sign: partial d match, eval intra_Pm_(Pdm_)

            if P.sublayers:
                new_sublayers = []
                for comb_subset_, subset_ in zip_longest(comb_sublayers, P.sublayers, fillvalue=([])):
                    # append combined subset_ (array of sub_P_ param sets):
                    new_sublayers.append(comb_subset_ + subset_)
                comb_sublayers = new_sublayers

                new_subDerts = []
                for comb_Dert, Dert in zip_longest(comb_subDerts, P.subDerts, fillvalue=([])):
                    new_Dert = []
                    if Dert or comb_Dert:  # at least one is not empty, from form_P_
                        new_Dert = [comb_param + param
                                   for comb_param, param in
                                   zip_longest(comb_Dert, Dert, fillvalue=0)]
                    new_subDerts.append(new_Dert)
                comb_subDerts = new_subDerts

    return comb_sublayers, comb_subDerts


def form_adjacent_M_(Pm_):  # compute array of adjacent Ms, for contrastive borrow evaluation
    '''
    Value is projected match, while variation has contrast value only: it matters to the extent that it interrupts adjacent match: adj_M.
    In noise, there is a lot of variation. but no adjacent match to cancel, so that variation has no predictive value.
    On the other hand, 2D outline or 1D contrast may have low gradient / difference, but it terminates some high-match span.
    Such contrast is salient to the extent that it can borrow predictive value from adjacent high-match area.
    adj_M is not affected by primary range_comp per Pm?
    no comb_m = comb_M / comb_S, if fid: comb_m -= comb_|D| / comb_S: alt rep cost
    same-sign comp: parallel edges, cross-sign comp: M - (~M/2 * rL) -> contrast as 1D difference?
    '''
    M_ = [0] + [Pm.M for Pm in Pm_] + [0]  # list of adj M components in the order of Pm_, + first and last M=0,

    adj_M_ = [ (abs(prev_M) + abs(next_M)) / 2  # mean adjacent Ms
               for prev_M, next_M in zip(M_[:-2], M_[2:])  # exclude first and last Ms
             ]
    ''' expanded:
    pri_M = Pm_[0].M  # deriv_comp value is borrowed from adjacent opposite-sign Ms
    M = Pm_[1].M
    adj_M_ = [abs(Pm_[1].M)]  # initial next_M, also projected as prior for first P
    for Pm in Pm_[2:]:
        next_M = Pm.M
        adj_M_.append((abs(pri_M / 2) + abs(next_M / 2)))  # exclude M
        pri_M = M
        M = next_M
    adj_M_.append(abs(pri_M))  # no / 2: projection for last P
    '''
    return adj_M_

def range_comp(dert_):  # cross-comp of 2**rng- distant pixels: 4,8,16.., skipping intermediate pixels
    rdert_ = []
    _i = dert_[0].i

    for dert in dert_[2::2]:  # all inputs are sparse, skip odd pixels compared in prior rng: 1 skip / 1 add to maintain 2x overlap
        d = dert.i -_i
        rp = dert.p + _i  # intensity accumulated in rng
        rd = dert.d + d   # difference accumulated in rng
        rm = dert.m + ave - abs(d)  # m accumulated in rng
        # for consistency with deriv_comp, else redundant
        rdert_.append( Cdert( i=dert.i,p=rp,d=rd,m=rm ))
        _i = dert.i

    return rdert_

def deriv_comp(dert_):  # cross-comp consecutive ds in same-sign dert_: sign match is partial d match
    # dd and md may match across d sign, but likely in high-match area, spliced by spec in comp_P?
    # initialization:
    ddert_ = []
    _d = abs( dert_[0].d)  # same-sign in Pd

    for dert in dert_[1:]:
        # same-sign in Pd
        d = abs( dert.d )
        rd = d + _d
        dd = d - _d
        md = min(d, _d) - abs( dd/2) - ave_min  # min_match because magnitude of derived vars corresponds to predictive value
        ddert_.append( Cdert( i=dert.d,p=rd,d=dd,m=md ))
        _d = d

    return ddert_


def form_rval_P_(iP_):

    # cluster Ps by the sign of value adjusted for cross-param redundancy,
    # re-evaluate them for cross-level rdn and consolidation: compact()
    rval_P__ = []
    Rval=0
    _sign = None  # to initialize 1st rdn P, (None != True) and (None != False) are both True

    for P in iP_:
        rval = P.M - 1 * ave_M * P.L
        sign = rval>0
        
        if sign != _sign:  # sign change, initialize rP and append it to rP_
            rval_P_ = [(rval, P)]
            Rval = rval
            if _sign:  # -rPs are not processed?
                compact_P_(rval_P_)  # re-eval Pps, Pp.pdert_s for redundancy, eval splice Ps
            rval_P__.append(( Rval, rval_P_))  # updated by accumulation below
        else:
            # accumulate params:
            Rval += rval
            rval_P_ += [(rval, P)]
        _sign = sign

    return rval_P__

# need further review:
def compact_P_(rval_P_):
    for i, (rval, P) in enumerate(rval_P_):
        # assign cross-level rdn (P vs. dert_), re-evaluate P and dert_:
        P_val = rval / P.L - ave  # / P.L: resolution reduction, but lower rdn:
        dert_val = rval - ave * P.L  # * P.L: ave cost * number of representations

        if P_val > dert_val: dert_val -= ave * 1
        else:                  P_val -= ave * 1  # ave scaled by rdn (rdn = 1)
        if P_val <= 0:
            rval_P_[i]  = (rval, CP(dert_=P.dert_))
        else:
            _P = CP() # not sure if need this?
            _P.dert_ = P.dert_
            for dert in P.dert_:
                _P.accum_from(dert)  # different from P params
                
            rval_P_.dert_[i] = (rval, _P) # replace P with spliced P,
            if dert_val <= 0:
                P.dert_ = []  # remove pdert_


if __name__ == "__main__":
    ''' 
    Parse argument (image)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='.//raccoon.jpg')
    arguments = vars(argument_parser.parse_args())
    # Read image
    image = cv2.imread(arguments['image'], 0).astype(int)  # load pix-mapped image
    '''
    fpickle = 2  # 0: load; 1: dump; 2: no pickling
    render = 0
    fline_PPs = 0
    start_time = time()
    
    if fpickle == 0:
        # Read frame_of_patterns from saved file instead
        with open("frame_of_patterns_.pkl", 'rb') as file:
            frame_of_patterns_ = pickle.load(file)
    else:
        # Run functions
        image = cv2.imread('.//toucan_small.jpg', 0).astype(int)  # manual load pix-mapped image
        assert image is not None, "No image in the path"
        # Main
        frame_of_patterns_ = cross_comp(image)  # returns Pm__
        if fpickle == 1: # save the dump of the whole data_1D to file
            with open("frame_of_patterns_.pkl", 'wb') as file:
                pickle.dump(frame_of_patterns_, file)

    if render:
        image = cv2.imread('.//raccoon.jpg', 0).astype(int)  # manual load pix-mapped image
        plt.figure(); plt.imshow(image, cmap='gray'); plt.show()  # show the image below in gray

    if fline_PPs:  # debug line_PPs
        from line_PPs import *
        
        frame_Pp__ = cross_comp_P_(frame_of_patterns_)
        draw_PP_(image, frame_Pp__)  # debugging
        
                
#     # recursive version
#    image = cv2.imread('.//raccoon.jpg', 0).astype(int)  # manual load pix-mapped image
#    feedback_H = []
#    cross_comp_(image, feedback_H, level=0, fPd=0, frecursive=1)
#    
#    end_time = time() - start_time
#    print(end_time)