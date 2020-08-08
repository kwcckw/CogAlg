'''
line_PPs is a 2nd level 1D algorithm.

It cross-compares line_patterns output Ps (s, L, I, D, M, dert_, layers) and evaluates them for deeper cross-comparison.
Depth of cross-comparison: range+ and deriv+, is increased in lower-recursion element_,
then between same-recursion element_s:

comp (s): if same-sign,
          cross-sign comp is borrow, also default L and M (core param) comp?
          discontinuous comp up to max rel distance +|- contrast borrow, with bi-directional selection?

    comp (L, I, D, M): equal-weight, select redundant I | (D,M),  div L if V_var * D_vars, and same-sign d_vars?
        comp (dert_):  lower composition than layers, if any
    comp (layers):  same-derivation elements
        comp (P_):  sub patterns

Increment of 2nd level alg over 1st level alg should be made recursive, forming relative-level meta-algorithm.

Comparison distance is extended to first match or maximal accumulated miss over compared dert_Ps, measured by roL*roM?
Match or miss may be between Ps of either sign, but comparison of lower P layers is conditional on higher-layer match

So, comparison between two Ps is variable-depth down P hierarchy, with sign at the top, until max accumulated miss.
(vertical induction: results of higher-layer comparison predict results of next-layer comparison)
This is similar to variable-range comp between Ps, until either first match or max miss (lateral induction).

Resulting PPs will be more like graphs (in 1D), with explicit distances between consecutive element Ps.
This is different from 1st level connectivity clustering, where all distances between consecutive elements = 1.
'''

ave_dI = 20
div_ave = 50
ave_Pm = 50
max_miss = 50

def comp_P(P_):

    dert_P_ = []  # array of Ps with added derivatives
    # _P = P_[0]  # 1st P  # oP = P_[1]  # 1st opposite-sign P

    for i, P in enumerate(P_, start=2):
        sign, L, I, D, M, dert_, sub_H = P
        roL = roM = 0
        rL = rM = 0

        for _P in (P_[i+1 :]):  # no past-P displacement, just shifting first _P for variable max-distance comp
            _sign, _L, _I, _D, _M, _dert_, _sub_H = _P

            roL += P_[i-1][1] / L  # relative distance, P_[i-1] is prior opposite-sign P
            roM += P_[i-1][4] / (abs(M)+1)  # relative miss or contrast: additional range limiter, roD if for dPP?
            # or ro_Pm: induction and search blocker is current-level match, not intra_input M?
            rL += P_[i-2][1] / L
            rM += P_[i-2][4] / (abs(M)+1)

            if roL*roM > max_miss:  # accumulated from all net-negative comparisons before first match
                dL = L - _L
                mL = min(L, _L)  # L: positions / sign, dderived: magnitude-proportional value
                dI = I - _I
                mI = abs(dI) - ave_dI
                dD = abs(D) - abs(_D)
                mD = min(abs(D), abs(_D))  # same-sign D in dP?
                dM = M - _M;  mM = min(M, _M)

                Pm = mL + mM + mD  # Pm *= roL * decay: contrast to global decay rate?
                ms = 1 if Pm > ave_Pm * 7 > 0 else 0  # comp cost = ave * 7, or rep cost: n vars per P?

                dert_P_.append( (ms, Pm, rL, rM, roL, roM, mL, dL, mI, dI, mD, dD, mM, dM, P))
                if ms:
                    break  # nearest-neighbour search, until first match, or:
            else:
                break  # reached maximal accumulated miss, stop search

    return dert_P_


def form_mPP(dert_P_):  # cluster dert_Ps by Pm sign

    mPP_ = []
    
    _ms, _Pm, _rL, _rM, _roL, _roM, _mL, _dL, _mI, _dI, _mD, _dD, _mM, _dM, _P = dert_P_[0]
    
    for ite, (ms, Pm, rL, rM, roL, roM, mL, dL, mI, dI, mD, dD, mM, dM, P) in enumerate(dert_P_,start=1):
        # in form_dPP:
        # Pd = dL + dM + dD  # -> directional dPP, equal-weight params, no rdn?
        # ds = 1 if Pd > 0 else 0

#    ''' evaluation for comp by division is per PP, not per P: results must be comparable between consecutive Ps  

        fdiv = 0
        nvars = []
        
        # pattern wide match + sum of ds?
        if Pm + abs(dL + dI + dD + dM) > div_ave:  
            
            # dI * rL is normalize I?
            norm_I = dI * rL
            _norm_I = _dI * _rL
            norm_dI = norm_I - _norm_I
            # how to compute normalize match from norm_dI?
        
            # this is pattern wide normalize match?
            Pnm = mX + nmDx + nmDy  # defines norm_mPP, no ndx: single, but nmx is summed

            # if pattern wide match > pattern wide normalize match, nmPP_rdn = 1, what is nmPP?
            if Pm > Pnm: nmPP_rdn = 1; mPP_rdn = 0  # added to rdn, or diff alt, olp, div rdn?
            else:        mPP_rdn = 1; nmPP_rdn = 0
            Pnd = ddX + ndDx + ndDy  # normalized d defines norm_dPP or ndPP
            # Pnd is pattern wide normalize difference?

            if Pd > Pnd: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
            else:        dPP_rdn = 1; ndPP_rdn = 0
            fdiv = 1
            nvars = Pnm, nmD, mPP_rdn, nmPP_rdn, Pnd, ndDx, ndDy, dPP_rdn, ndPP_rdn

        else:
            fdiv = 0  # DIV comp flag
            nvars = 0  # DIV + norm derivatives
            
 
        # update
        _ms = ms
        _Pm = Pm
        _rL = rL
        _rM = rM
        _roL = roL
        _roM = roM
        _mL = mL
        _dL = dL
        _mI = mI
        _dI = dI
        _mD = mD
        _dD = dD
        _mM = mM
        _dM = dM
        _P = P    

    return mPP_
