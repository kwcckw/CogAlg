
"""
Classes within slice operations
"""
from class_cluster import ClusterStructure, NoneType

## slice_blob ##

class CP(ClusterStructure):
    I = int
    Dy = int
    Dx = int
    G = int
    M = int
    Dyy = int
    Dyx = int
    Dxy = int
    Dxx = int
    Ga = int
    Ma = int
    L = int
    x0 = int
    sign = NoneType
    dert_ = list
    gdert_ = list
    Dg = int
    Mg = int

class CStack(ClusterStructure):
    # Dert:
    I = int
    Dy = int
    Dx = int
    G = int
    M = int
    Dyy = int
    Dyx = int
    Dxy = int
    Dxx = int
    Ga = int
    Ma = int
    A = int  # stack area
    Ly = int
    # box:
    x0 = int
    xn = int
    y0 = int
    # stack:
    Py_ = list  # Py_, dPPy_, or stack_
    sign = NoneType
    f_gstack = NoneType  # gPPy_ if 1, else Py_
    f_stackPP = NoneType  # for comp_slice: PPy_ if 1, else gPPy_ or Py_
    downconnect_cnt = int
    upconnect_ = list
    stack_PP = object  # replaces f_stackPP?
    stack_ = list  # ultimately all stacks, also replaces fflip: vertical if empty, else horizontal
    f_checked = int  # flag: stack has gone through form_sstack_recursive as upconnect


## comp_slice ##
    
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

class CPP(ClusterStructure):

    dert_Pi = object  # PP params = accumulated dert_P params:
    # PM, PD, MX, DX, ML, DL, MDx, DDx, MDy, DDy, MDg, DDg, MMg, DMg; also accumulate P params?
    dert_P_ = list   # only refs to stack_PP dert_Ps
    fmPP = NoneType  # mPP if 1, else dPP; not needed if packed in PP_?
    fdiv = NoneType  # or defined per stack?

class CStack_PP(ClusterStructure):

    dert_Pi = object  # stack_PP params = accumulated dert_P params:
    # sPM, sPD, sMX, sDX, sML, sDL, sMDx, sDDx, sMDy, sDDy, sMDg, sDDg, sMMg, sDMg
    mPP_ = list
    dPP_ = list
    dert_P_ = list
    fdiv = NoneType




