'''
    comp_slice_ is a terminal fork of intra_blob.
    It traces blob axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
    These high-G high-Ma blobs are vectorized into outlines of adjacent flat or low-G blobs.
    Vectorization is clustering of Ps + derivatives into PPs: patterns of Ps that describe an edge.
    Double edge lines: assumed match between edges of high-deviation intensity, no need for cross-comp?
    secondary cross-comp of low-deviation blobs?   P comb -> intra | inter comp eval?
    radial comp extension for co-internal blobs:
    != sign comp x sum( adj_blob_) -> intra_comp value, isolation value, cross-sign merge if weak, else:
    == sign comp x ind( adj_adj_blob_) -> same-sign merge | composition:
    borrow = adj_G * rA: default sum div_comp S -> relative area and distance to adjj_blob_
    internal sum comp if mA: in thin lines only? comp_norm_G or div_comp_G -> rG?
    isolation = decay + contrast:
    G - G * (rA * ave_rG: decay) - (rA * adj_G: contrast, = lend | borrow, no need to compare vG?)
    if isolation: cross adjj_blob composition eval,
    else:         cross adjj_blob merge eval:
    blob merger if internal match (~raG) - isolation, rdn external match:
    blob compos if external match (~rA?) + isolation,
    Also eval comp_slice over fork_?
    rng+ should preserve resolution: rng+_dert_ is dert layers,
    rng_sum-> rng+, der+: whole rng, rng_incr-> angle / past vs next g,
    rdn Rng | rng_ eval at rng term, Rng -= lost coord bits mag, always > discr?
'''

from time import time
from collections import deque
from class_cluster import ClusterStructure, NoneType
from math import hypot
import numpy as np
import warnings  # to detect overflow issue, in case of infinity loop
warnings.filterwarnings('error')

ave = 20
div_ave = 200
flip_ave = 1000
ave_dX = 10  # difference between median x coords of consecutive Ps

class Cdert_P(ClusterStructure):

    Pi = object  # P instance, accumulation: Cdert_P.Pi.I += 1, etc.
    Pm = int
    Pd = int
    mx = int
    dx = int
    mL = int
    dL = int
    mDx = int
    dDx = int
    mDy = int
    dDy = int
    mDg = int
    dDg = int
    mMg = int
    dMg = int
    upconnect_ = list
    downconnect_cnt = int
    # stack and PP object reference
    stack = object 
    PP = object

class CPP(ClusterStructure):

    dert_P_ = list
    dert_Pi = object
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_?
    fdiv = NoneType
    # between PPs:
    upconnect_ = list
    downconnect_cnt = int
    upconnect_cnt = int


def comp_slice_(stack_, _P):
    '''
    cross-compare connected Ps of stack_, including Ps of adjacent stacks (upconnects)
    '''
    dert_P_ = []
    DdX = 0
    for stack in reversed(stack_):  # bottom-up

        if not stack.f_checked :  # else this stack has been scanned as some other upconnect
            stack.f_checked = 1
            downconnect_cnt = stack.downconnect_cnt

            if not _P:  # stack is from blob.stack_
                _P = stack.Py_.pop()
                # 1st dert_P will not run through comp_slice
                dert_P_.append(Cdert_P(Pi=_P, downconnect_cnt = downconnect_cnt, stack = stack))
                # assign stack.downconnect_cnt to dert_P.downconnect_cnt, no derivatives in 1st dert_P

            for P in reversed(stack.Py_):
                dert_P = comp_slice(P, _P, DdX)  # ortho and other conditional operations are evaluated per PP
                dert_P.downconnect_cnt = downconnect_cnt
                if dert_P_:
                    dert_P_[-1].upconnect_.append(dert_P)  # next
                dert_P_.append(dert_P)  # dert_P is converted to Cdert_P in comp_slice
                _P = P
                downconnect_cnt = 1  # always 1 inside Py_

#            dert_P_[-1].upconnect_ = stack.upconnect_

            if stack.upconnect_:
                dert_P_ += comp_slice_(stack.upconnect_, _P)  # recursive compare _P to all upconnected Ps
                
    return dert_P_


def comp_slice(P, _P, DdX):  # forms vertical derivatives of P params, and conditional ders from norm and DIV comp

    s, x0, G, M, Dx, Dy, L, Dg, Mg = P.sign, P.x0, P.G, P.M, P.Dx, P.Dy, P.L, P.Dg, P.Mg
    # params per comp branch, add angle params, ext: X, new: L,
    # no input I comp in top dert?
    _s, _x0, _G, _M, _Dx, _Dy, _L, _Dg, _Mg = _P.sign, _P.x0, _P.G, _P.M, _P.Dx, _P.Dy, _P.L, _P.Dg, _P.Mg
    '''
    redefine Ps by dx in dert_, rescan dert by input P d_ave_x: skip if not in blob?
    '''
    xn = x0 + L-1;  _xn = _x0 + _L-1
    mX = min(xn, _xn) - max(x0, _x0)  # overlap: abs proximity, cumulative binary positional match | miss:
    _dX = (xn - L/2) - (_xn - _L/2)
    dX = abs(x0 - _x0) + abs(xn - _xn)  # offset, or max_L - overlap: abs distance?

    if dX > ave_dX:  # internal comp is higher-power, else two-input comp not compressive?
        rX = dX / (mX+.001)  # average dist/prox, | prox/dist, | mX / max_L?

    ave_dx = (x0 + (L-1)//2) - (_x0 + (_L-1)//2)  # d_ave_x, median vs. summed, or for distant-P comp only?

    ddX = dX - _dX  # long axis curvature
    DdX += ddX  # if > ave: ortho eval per P, else per PP_dX?
    # param correlations: dX-> L, ddX-> dL, neutral to Dx: mixed with anti-correlated oDy?
    '''
    if ortho:  # estimate params of P locally orthogonal to long axis, maximizing lateral diff and vertical match
        Long axis is a curve, consisting of connections between mid-points of consecutive Ps.
        Ortho virtually rotates each P to make it orthogonal to its connection:
        hyp = hypot(dX, 1)  # long axis increment (vertical distance), to adjust params of orthogonal slice:
        L /= hyp
        # re-orient derivatives by combining them in proportion to their decomposition on new axes:
        Dx = (Dx * hyp + Dy / hyp) / 2  # no / hyp: kernel doesn't matter on P level?
        Dy = (Dy / hyp - Dx * hyp) / 2  # estimated D over vert_L
    '''
    dL = L - _L; mL = min(L, _L)  # L: positions / sign, dderived: magnitude-proportional value
    dM = M - _M; mM = min(M, _M)  # no Mx, My: non-core, lesser and redundant bias?

    dDx = abs(Dx) - abs(_Dx); mDx = min(abs(Dx), abs(_Dx))  # same-sign Dx in vxP
    dDy = Dy - _Dy; mDy = min(Dy, _Dy)  # Dy per sub_P by intra_comp(dx), vs. less vertically specific dI

    # gdert param comparison, if not fPP, values would be 0
    dMg = Mg - _Mg; mMg = min(Mg, _Mg)
    dDg = Dg - _Dg; mDg = min(Dg, _Dg)

    Pd = ddX + dL + dM + dDx + dDy + dMg + dDg # -> directional dPP, equal-weight params, no rdn?
    # correlation: dX -> L, oDy, !oDx, ddX -> dL, odDy ! odDx? dL -> dDx, dDy?  G = hypot(Dy, Dx) for 2D structures comp?
    Pm = mX + mL + mM + mDx + mDy + mMg + mDg # -> complementary vPP, rdn *= Pd | Pm rolp?

    dert_P = Cdert_P(Pi=P, Pm=Pm, Pd=Pd, mX=mX, dX=dX, mL=mL, dL=dL, mDx=mDx, dDx=dDx, mDy=mDy, dDy=dDy, mDg=mDg, dDg=dDg, mMg=mMg, dMg=dMg)
    # div_f, nvars

    return dert_P


def accum_PP(PP, dert_P):  # accumulate PP

    # accumulate sstack params into PP
    PP.dert_Pi.accumulate(Pm=dert_P.Pm, Pd=dert_P.Pd, mx=dert_P.mx, dx=dert_P.dx,
                          mL=dert_P.mL, dL=dert_P.dL, mDx=dert_P.mDx, dDx=dert_P.dDx,
                          mDy=dert_P.mDy, dDy=dert_P.dDy, mDg=dert_P.mDg, dDg=dert_P.dDg,
                          mMg=dert_P.mMg, dMg=dert_P.dMg)

def dert_P_2_PP_(dert_P_, PP, PP_):
    '''
    just a draft,
    first row of dert_P_ has downconnect_cnt == 0, higher rows may also have them
    '''
    
    if not PP: # from dert_P_
        
        re_init = 0
        _dert_P = dert_P_[0]
        PP = CPP(dert_Pi=_dert_P, dert_P_= [_dert_P])
        
        for i, dert_P in enumerate(dert_P_):  # bottom-up to follow upconnects
            if dert_P.downconnect_cnt == 0:  # root dert_Ps were not checked, upconnects always checked
                # i think the section below is not necessary , since each dert_P with downconnect_cnt == 0 should start 1 new PP
                # unless those PP should be connected as well
                
                if re_init:
                    PP = CPP(dert_Pi=dert_P, dert_P_= [dert_P])
                
                elif (_dert_P.Pm > 0) != (dert_P.Pm > 0): # check for sin change only if both prior and current dert_P are base dert_P
                    PP_.append(PP)
                    PP = CPP(dert_Pi=Cdert_P(), dert_P_= [dert_P])
                    dert_P.PP = PP
                    accum_PP(PP, dert_P)  # regardless of termination
                    
                _dert_P = dert_P
                dert_P_2_PP_(dert_P.upconnect_, PP, PP_ )  # form PPs across upconnects
            else:
                re_init = 1 # reinitialize PP if prior dert_P is base dert_P, while current one is not

    else: # from recursive loop  

        if PP.upconnect_cnt: PP.upconnect_cnt -= 1 
            
        upconnect_ = []
        if dert_P_: # there is upconnects and PP from prior function
            _dert_P = PP.dert_P_[-1]
            for i, dert_P in enumerate(dert_P_):
                if (_dert_P.Pm > 0) != (dert_P.Pm > 0):
                    PP.upconnect_.append(dert_P_.pop(i))

        PP.upconnect_cnt += len(upconnect_) 
        
        if len(PP.upconnect_)>0: # if there is any upconnect for current PP

            _dert_P = PP.dert_P_[-1]
            for i, dert_P in enumerate(upconnect_):  # bottom-up to follow upconnects
                if not isinstance(dert_P.PP, CPP):  # checked dert_P should have PP object instance
                    if (_dert_P.Pm > 0) != (dert_P.Pm > 0): 
                        PP = CPP(dert_Pi=Cdert_P(), dert_P_= [dert_P])
                        dert_P.PP = PP
                        
                    accum_PP(PP, dert_P)  # regardless of termination
                    dert_P_2_PP_(dert_P.upconnect_, PP, PP_ )  # form PPs across upconnects
        
        else: # terminate PP when upconnect = 0, should be at the end of the upconnect
            if PP.upconnect_cnt == 0:
                PP_.append(PP)

        # form PP again with the non upconnect's dert_Ps
        for dert_P in dert_P_: 
            PP = CPP(dert_Pi=Cdert_P(), dert_P_= [dert_P]) # form new PP for new non upconnect's dert_Ps
            dert_P_2_PP_(dert_P.upconnect_, PP, PP_)

