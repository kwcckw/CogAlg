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

class CPP(ClusterStructure):

    stack_ = list
    stack_PP_ = list
    # possibly multiple between PPs
    upconnect_ = list
    downconnect_cnt = int
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_?
    fdiv = NoneType


class CStack_PP(ClusterStructure):
    # needs to be revised, this is now an element of stack_ in PP,
    upconnect_ = list
    downconnect_cnt = int
    #+ other params of CStack?
    dert_Py_ = list
    fdiv = NoneType
    dert_Pi = object  # stack_PP params = accumulated dert_P params:
    # sPM, sPD, sMX, sDX, sML, sDL, sMDx, sDDx, sMDy, sDDy, sMDg, sDDg, sMMg, sDMg
    # add to blob:
    # PPm_ = list
    # PPd_ = list  # these are now primary blob-level structures


def comp_slice_(stack_, _P):
    '''
    cross-compare connected Ps of stack_, including Ps of adjacent stacks (upconnects)
    '''
    for stack in reversed(stack_):

        if not stack.f_checked :
            stack.f_checked = 1
            DdX = 0  # this may also have to be accumulated across stacks

            if _P:  # stack_ = upconnect_
                dert_Py_ = [Cdert_P(Pi=_P)]  # initialize dert_P with _P only, other params are 0
            else:  # stack_ = blob.stack_, no prior P
                dert_Py_ = []
                _P = stack.Py_.pop()

            for P in reversed(stack.Py_):
                dert_P = comp_slice(P, _P, DdX)  # ortho and other conditional operations are evaluated per PP
                dert_Py_.append(dert_P)  # dert_P should be converted to Cdert_P in comp_slice
                _P = P

            # when stack is having single P, dert_Py_ is empty so adding Pi = _P
            if not dert_Py_:
                dert_Py_ = [Cdert_P(Pi=_P)]

            stack.Py_ = dert_Py_

            # recursive compare _P to all upconnected Ps
            comp_slice_(stack.upconnect_, _P)


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

    dert_P = Cdert_P(Pi=_P, Pm=Pm, Pd=Pd, mX=mX, dX=dX, mL=mL, dL=dL, mDx=mDx, dDx=dDx, mDy=mDy, dDy=dDy, mDg=mDg, dDg=dDg, mMg=mMg, dMg=dMg)
    # div_f, nvars

    return dert_P


def comp_slice_old(blob, AveB):  # comp_slice eval per blob, simple stack_

        for stack in blob.stack_:
            if stack.G * stack.Ma - AveB / 10 > 0:  # / 10: ratio AveB to AveS, or not needed?

                # or default (L, Dy, Dx, G) min comp for rotation,
                # primary comp L, the rest is normalized?
                # overlap vs. shift:
                # init cross-dimension div_comp: Dx/Dy, but separate val for comp G, no default G/L?
                # comp -> min Dx/Dy for rotation, min G for comp_g?

                # also default min comp to upconnect_ Ps -> forking / merging PPs -> stack_ per PP!
                # stack.f_stackPP = 1  # scan Py_ -> comp_slice -> form_PP -> 2D PP_: clusters of same-sign dP | mP
                DdX = 0

                if stack.G * (stack.Ly / stack.A) * (abs(stack.Dy) / abs((stack.Dx) + 1)) > ave:  # G_bias * L_bias -> virt.rotation:
                    # or default min comp, eval per PP?
                    ortho = 1  # estimate params of P orthogonal to long axis at P' y and ave_x, to increase mP
                else:
                    ortho = 0
                dert_P_ = []
                _P = stack.Py_[0]

                for P in stack.Py_[1:]:
                    dert_P = comp_slice(ortho, P, _P, DdX)
                    dert_P_.append(dert_P)
                    _P = P


def form_PP_(stack_, PP_, PP, stack_PP, _dert_P):  # terminate, initialize, increment PPs

    # initial draft to test on the concept, buggy and not complete yet
    # cluster all connected dert_Ps of same-sign mP
    for stack in reversed(stack_):

        if stack.f_checked: # stack.f_checked value is equal to 1 after comp_slice_
            stack.f_checked = 0
            if _dert_P:  # stack_ = upconnect_  # stack_PP = PP.stack_PP_[-1]; _dert_P = stack_PP.dert_Py_[-1]
                i = 0
            else:  # stack_ = blob.stack_
                _dert_P = stack.Py_[0]
                i = 1

            for dert_P in stack.Py_[i:]:
                # cluster dert_Ps into PPs
                if _dert_P.Pm > 0 != dert_P.Pm > 0:  # sign change

                    if PP and stack_PP:
                        PP.stack_PP_.append(stack_PP)  # pack stack_PP
                    else:
                        stack_PP_ = [ CStack_PP(dert_Py_=[_dert_P]) ]
                        PP = CPP(stack_PP_=stack_PP_)
                    stack_PP = CStack_PP(dert_Pi=Cdert_P())  # reinitialize

                accum_stack_PP(stack_PP, _dert_P)  # pack _dert_P into stack_PP, regardless of termination
                _dert_P = dert_P

            PP.stack_PP_.append(stack_PP)
            # recursively cluster stack upconnects:
            form_PP_(stack.upconnect_, PP_, PP, stack_PP, _dert_P)

            # temporary fix, to avoid duplicated PP
            if PP not in PP_:
                PP_.append(PP)  # PP is terminated after scanning through all upconnects
            PP = []  # we need this to terminate PP

    return PP_


def accum_gstack(gstack_PP, istack, stack_PP):   # accumulate istack and stack_PP into stack
    '''
    This looks wrong, accum_nested_stack should be an add-on to accum_stack_PP
    only called if istack.f_stack_PP:
    accum_stack_PP accumulates dert_P into stack_PP
    # accum_gstack accumulates stack_PP into gstack_PP?
    '''

    if istack.f_stack_PP:  # input stack is stack_PP
        # stack_PP params
        dert_Pi, mPP_, dPP_, dert_P_, fdiv = stack_PP.unpack()

        # need to accumulate dert_P params here, from stack_PP.dert_P params
        # accumulate stack_PP params
        gstack_PP.stack_PP.mPP_.extend(mPP_)
        gstack_PP.stack_PP.dPP_.extend(dPP_)
        gstack_PP.stack_PP.dert_P_.extend(dert_P_)
        gstack_PP.stack_PP.fdiv = fdiv

    # istack params
    I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, A, Ly, x0, xn, y0, Py_, sign, _, _, _, _, _, _, _  = istack.unpack()

    # accumulate istack param into stack_stack_PP
    gstack_PP.I += I
    gstack_PP.Dy += Dy
    gstack_PP.Dx += Dx
    gstack_PP.G += G
    gstack_PP.M += M
    gstack_PP.Dyy += Dyy
    gstack_PP.Dyx += Dyx
    gstack_PP.Dxy += Dxy
    gstack_PP.Dxx += Dxx
    gstack_PP.Ga += Ga
    gstack_PP.Ma += Ma
    gstack_PP.A += A
    gstack_PP.Ly += Ly
    if gstack_PP.x0 > x0: gstack_PP.x0 = x0
    if gstack_PP.xn < xn: gstack_PP.xn = xn
    if gstack_PP.y0 > y0: gstack_PP.y0 = y0
    gstack_PP.Py_.extend(Py_)
    gstack_PP.sign = sign  # sign should be same across istack


def accum_stack_PP(stack_PP, dert_P):  # accumulate mPPs or dPPs

    _, Pm, Pd, mx, dx, mL, dL, mDx, dDx, mDy, dDy, mDg, dDg, mMg, dMg = dert_P.unpack()

    # accumulate dert_P params into stack_PP
    stack_PP.dert_Pi.Pm += Pm
    stack_PP.dert_Pi.Pd += Pd
    stack_PP.dert_Pi.mx += mx
    stack_PP.dert_Pi.dx += dx
    stack_PP.dert_Pi.mL += mL
    stack_PP.dert_Pi.dL += dL
    stack_PP.dert_Pi.mDx += mDx
    stack_PP.dert_Pi.dDx += dDx
    stack_PP.dert_Pi.mDy += mDy
    stack_PP.dert_Pi.dDy += dDy
    stack_PP.dert_Pi.mDg += mDg
    stack_PP.dert_Pi.dDg += dDg
    stack_PP.dert_Pi.mMg += mMg
    stack_PP.dert_Pi.dMg += dMg

    stack_PP.dert_Py_.append(dert_P)


def accum_PP(dert_P, PP):  # accumulate mPPs or dPPs

    # dert_P params
    _, Pm, Pd, mx, dx, mL, dL, mDx, dDx, mDy, dDy, mDg, dDg, mMg, dMg = dert_P.unpack()

    # accumulate dert_P params into PP
    PP.dert_Pi.Pm += Pm
    PP.dert_Pi.Pd += Pd
    PP.dert_Pi.mx += mx
    PP.dert_Pi.dx += dx
    PP.dert_Pi.mL += mL
    PP.dert_Pi.dL += dL
    PP.dert_Pi.mDx += mDx
    PP.dert_Pi.dDx += dDx
    PP.dert_Pi.mDy += mDy
    PP.dert_Pi.dDy += dDy
    PP.dert_Pi.mDg += mDg
    PP.dert_Pi.dDg += dDg
    PP.dert_Pi.mMg += mMg
    PP.dert_Pi.dMg += dMg
    PP.dert_Py_.append(dert_P)

'''
    Pd and Pm are ds | ms per param summed in P. Primary comparison is by subtraction, div if par * rL compression:
    DL * DS > min: must be both, eval per dPP PD, signed? comp d?
    - resulting vertically adjacent dPPs and vPPs are evaluated for cross-comparison, to form PPPs and so on
    - resulting param derivatives form par_Ps, which are evaluated for der+ and rng+ cross-comparison
    | default top+ P level: if PD | PM: add par_Ps: sub_layer, rdn ele_Ps: deeper layer?
    aS compute if positive eV (not qD?) = mx + mL -ave? :
    aI = I / L; dI = aI - _aI; mI = min(aI, _aI)
    aD = D / L; dD = aD - _aD; mD = min(aD, _aD)
    aM = M / L; dM = aM - _aM; mM = min(aM, _aM)
    d_aS comp if cs D_aS, iter dS - S -> (n, M, diff): var precision or modulo + remainder?
    pP_ eval in +vPPs only, per rdn = alt_rdn * fork_rdn * norm_rdn, then cost of adjust for pP_rdn?
    eval_div(PP):
    if dL * Pm > div_ave:  # dL = potential compression by ratio vs diff, or decremental to Pd and incremental to Pm?
        rL  = L / _L  # DIV comp L, SUB comp (summed param * rL) -> scale-independent d, neg if cross-sign:
        nDx = Dx * rL; ndDx = nDx - _Dx; nmDx = min(nDx, _Dx)  # vs. nI = dI * rL or aI = I / L?
        nDy = Dy * rL; ndDy = nDy - _Dy; nmDy = min(nDy, _Dy)
        Pnm = mX + nmDx + nmDy  # defines norm_mPP, no ndx: single, but nmx is summed
        if Pm > Pnm: nmPP_rdn = 1; mPP_rdn = 0  # added to rdn, or diff alt, olp, div rdn?
        else: mPP_rdn = 1; nmPP_rdn = 0
        Pnd = ddX + ndDx + ndDy  # normalized d defines norm_dPP or ndPP
        if Pd > Pnd: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
        else: dPP_rdn = 1; ndPP_rdn = 0
        div_f = 1
        nvars = Pnm, nmDx, nmDy, mPP_rdn, nmPP_rdn,  Pnd, ndDx, ndDy, dPP_rdn, ndPP_rdn
    else:
        div_f = 0  # DIV comp flag
        nvars = 0  # DIV + norm derivatives
    '''

def term_PP(typ, PP):  # eval for orient (as term_blob), incr_comp_slice, scan_par_:

    s, L2, I2, D2, Dy2, M2, My2, G2, Olp2, Py_, PM, PD, Mx, Dx, ML, DL, MI, DI, MD, DD, MDy, DDy, MM, DM, MMy, DMy, nVars = PP

    rdn = Olp2 / L2  # rdn per PP, alt Ps (if not alt PPs) are complete?

    # if G2 * Dx > ave * 9 * rdn and len(Py_) > 2:
    # PP, norm = orient(PP) # PP norm, rescan relative to parent blob, for incr_comp, comp_sliceP, and:

    if G2 + PM > ave * 99 * rdn and len(Py_) > 2:
       PP = incr_range_comp_slice(typ, PP)  # forming incrementally fuzzy PP

    if G2 + PD > ave * 99 * rdn and len(Py_) > 2:
       PP = incr_deriv_comp_slice(typ, PP)  # forming incrementally higher-derivation PP

    if G2 + PM > ave * 99 * rdn and len(Py_) > 2:  # PM includes results of incr_comp_slice
       PP = scan_params(0, PP)  # forming vpP_ and S_p_ders

    if G2 + PD > ave * 99 * rdn and len(Py_) > 2:  # PD includes results of incr_comp_slice
       PP = scan_params(1, PP)  # forming dpP_ and S_p_ders

    return PP

''' incr_comp() ~ recursive_comp() in line_POC(), with Ps instead of pixels?
    with rescan: recursion per p | d (signed): frame(meta_blob | blob | PP)? '''

def incr_range_comp_slice(typ, PP):
    return PP

def incr_deriv_comp_slice(typ, PP):
    return PP

def scan_params(typ, PP):  # at term_network, term_blob, or term_PP: + P_ders and nvars?
    '''
    Aves (integer filters) and coefs (ratio filters) per parameter type trigger formation of parameter_Ps,
    after full-blob comp_slice_ sums match and miss per parameter.
    Also coefs per sub_blob from comp_blob_: potential parts of a higher object?
    '''
    P_ = PP[11]
    Pars = [ (0,0,0,[]) ]

    for P in P_:  # repack ders into par_s by parameter type:

        s, ix, x, I, D, Dy, M, My, G, oG, Olp, t2_, Pm, Pd, mx, dx, mL, dL, mI, dI, mD, dD, mDy, dDy, mM, dM, mMy, dMy, div_f, nvars = P
        pars_ = [(x, mx, dx), (len(t2_), mL, dL), (I, mI, dI), (D, mD, dD), (Dy, mDy, dDy), (M, mM, dM), (My, mMy, dMy)]  # no nvars?

        for par, Par in zip(pars_, Pars): # PP Par (Ip, Mp, Dp, par_) += par (p, mp, dp):

            p, mp, dp = par
            Ip, Mp, Dp, par_ = Par

            Ip += p; Mp += mp; Dp += dp; par_.append((p, mp, dp))
            Par = Ip, Mp, Dp, par_  # how to replace Par in Pars_?

    for Par in Pars:  # select form_par_P -> Par_vP, Par_dP: combined vs. separate: shared access and overlap eval?
        Ip, Mp, Dp, par_ = Par

        if Mp + Dp > ave * 9 * 7 * 2 * 2:  # ave PP * ave par_P rdn * rdn to PP * par_P typ rdn?
            par_vPS, par_dPS = form_par_P(0, par_)
            par_Pf = 1  # flag
        else:
            par_Pf = 0; par_vPS = Ip, Mp, Dp, par_; par_dPS = Ip, Mp, Dp, par_

        Par = par_Pf, par_vPS, par_dPS
        # how to replace Par in Pars_?

    return PP

def form_par_P(typ, param_):  # forming parameter patterns within par_:

    p, mp, dp = param_.pop()  # initial parameter
    Ip = p, Mp = mp, Dp = dp, p_ = []  # Par init

    _vps = 1 if mp > ave * 7 > 0 else 0  # comp cost = ave * 7, or rep cost: n vars per par_P?
    _dps = 1 if dp > 0 else 0

    par_vP = Ip, Mp, Dp, p_  # also sign, typ and par olp: for eval per par_PS?
    par_dP = Ip, Mp, Dp, p_
    par_vPS = 0, 0, 0, []  # IpS, MpS, DpS, par_vP_
    par_dPS = 0, 0, 0, []  # IpS, MpS, DpS, par_dP_

    for par in param_:  # all vars are summed in incr_par_P
        p, mp, dp = par
        vps = 1 if mp > ave * 7 > 0 else 0
        dps = 1 if dp > 0 else 0

        if vps == _vps:
            Ip, Mp, Dp, par_ = par_vP
            Ip += p; Mp += mp; Dp += dp; par_.append(par)
            par_vP = Ip, Mp, Dp, par_
        else:
            par_vP = term_par_P(0, par_vP)
            IpS, MpS, DpS, par_vP_ = par_vPS
            IpS += Ip; MpS += Mp; DpS += Dp; par_vP_.append(par_vP)
            par_vPS = IpS, MpS, DpS, par_vP_
            par_vP = 0, 0, 0, []

        if dps == _dps:
            Ip, Mp, Dp, par_ = par_dP
            Ip += p; Mp += mp; Dp += dp; par_.append(par)
            par_dP = Ip, Mp, Dp, par_
        else:
            par_dP = term_par_P(1, par_dP)
            IpS, MpS, DpS, par_dP_ = par_dPS
            IpS += Ip; MpS += Mp; DpS += Dp; par_dP_.append(par_dP)
            par_vPS = IpS, MpS, DpS, par_dP_
            par_dP = 0, 0, 0, []

        _vps = vps; _dps = dps

    return par_vPS, par_dPS  # tuples: Ip, Mp, Dp, par_P_, added to Par

    # LIDV per dx, L, I, D, M? also alt2_: fork_ alt_ concat, for rdn per PP?
    # fpP fb to define vpPs: a_mx = 2; a_mw = 2; a_mI = 256; a_mD = 128; a_mM = 128

def term_par_P(typ, par_P):  # from form_par_P: eval for orient, re_comp? or folded?
    return par_P

def scan_par_P(typ, par_P_):  # from term_PP, folded in scan_par_? pP rdn per vertical overlap?
    return par_P_

def comp_slice_P(par_P, _par_P):  # with/out orient, from scan_pP_
    return par_P

def scan_PP_(PP_):  # within a blob, also within a segment?
    return PP_

def comp_sliceP(PP, _PP):  # compares PPs within a blob | segment, -> forking PPP_: very rare?
    return PP

'''  
    horiz_dim_val = ave_Lx - |Dx| / 2  # input res and coord res are adjusted so mag approximates predictive value,
    vertical_dim_val  = Ly - |Dy| / 2  # or proj M = M - (|D| / M) / 2: no neg? 
    
    core params G and M represent value of all others, no max Pm = L + |V| + |Dx| + |Dy|: redundant and coef-filtered?
    no * Ave_blob / Ga: angle match rate, already represented by hforks' position + mag' V+G -> comp( d | ortho_d)?   
    eval per blob, too expensive for seg? no abs_Dx, abs_Dy for comp dert eval: mostly redundant?
    
    colors will be defined as color / sum-of-colors, color Ps are defined within sum_Ps: reflection object?
    relative colors may match across reflecting objects, forming color | lighting objects?     
    comp between color patterns within an object: segmentation? 
'''