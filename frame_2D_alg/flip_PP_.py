'''
Descriptions
'''
from class_cluster import ClusterStructure, NoneType

class CflipPP(ClusterStructure):

    derPP = object  # set of derP params accumulated in PP
    derP_ = list
    # between PPs:
    upconnect_ = list
    downconnect_cnt = int
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_?
    fdiv = NoneType
    PP_ = list

def form_flipPP(PP_, flipPP_):
    
    for PP in PP_:
        if not PP.f_checked: # PP is not checked yet
            PP.f_checked = 1
            
            flipPP = CflipPP(derPP=PP.derPP, derP_=PP.derP_, PP_=[PP]) # init 1st flipPP
            
            for derP in PP.derP_:  # search through connectivity via upconnects of derP._P
                upconnect_2_flipPP(derP, flipPP, flipPP_)
                
            flipPP_.append(flipPP) # terminate flipPP after scanning all upconnects


def upconnect_2_flipPP(derP, iflipPP, flipPP_):
    
    flipPP = iflipPP
    for _derP in derP._P.upconnect_: # _derP is upconnect's derP
        if not _derP.PP.f_checked: # _derP.PP is not checked yet
            _derP.PP.f_checked = 1
            
            horizontal_bias = compute_h_bias(_derP.PP)
            
            # 3 conditions
            # - horizontal bias > 1
            # - upconnect's PP not in current flipPP yet
            # - upconnect's PP is not associated to any existing flipPP yet
            if horizontal_bias and _derP.PP not in flipPP.PP_ and not isinstance(_derP.PP.flipPP, CflipPP):
                accum_flipPP(_derP.PP, flipPP) # accumulate PP into flipPP
            
            elif not isinstance(_derP.PP.flipPP, CflipPP): # _derP.PP is not associated with flipPP yet
                flipPP = CflipPP(derPP=_derP.PP.derPP, derP_=_derP.PP.derP_, PP_=[_derP.PP])
                
            # recursively form flipPP via upconnect
            if _derP._P.upconnect_:
                upconnect_2_flipPP(_derP, flipPP, flipPP_)
            elif flipPP is not iflipPP: # terminate new initialized flipPP in current function call
                flipPP_.append(flipPP)      


def accum_flipPP(PP, flipPP):
    
    flipPP.derPP.accumulate(Pm=PP.derPP.Pm, Pd=PP.derPP.Pd, mx=PP.derPP.mx, dx=PP.derPP.dx, mL=PP.derPP.mL, dL=PP.derPP.dL, mDx=PP.derPP.mDx, dDx=PP.derPP.dDx,
                            mDy=PP.derPP.mDy, dDy=PP.derPP.dDy, mDg=PP.derPP.mDg, dDg=PP.derPP.dDg, mMg=PP.derPP.mMg, dMg=PP.derPP.dMg)
    flipPP.PP_.append(PP)
    PP.flipPP = flipPP


def compute_h_bias(PP):
    
    # compute xn, x0 and Ly
    x0_, xn_, y_ = [], [], []
    for derP in PP.derP_:
        x0_.append(derP._P.x0) # take only P instead of _P & P?
        x0_.append(derP.P.x0)
        xn_.append(derP._P.x0+derP._P.L)
        xn_.append(derP.P.x0+derP.P.L)
        y_.append(derP._P.y)
        y_.append(derP.P.y)
    x0 = min(x0_)
    xn = min(xn_)
    Ly = max(y_) - min(y_)
    
    # ratio = average of (mDy/mDx) & (dDy/dDx) ?
    dydx_ratio = ( (abs(PP.derPP.mDy) / abs(PP.derPP.mDx)) + (abs(PP.derPP.dDy) / abs(PP.derPP.dDx))  )/2
    
    horizontal_bias = ((xn - x0) / Ly) * dydx_ratio
    
    return horizontal_bias