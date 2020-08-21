
from itertools import zip_longest

'''
line_PPs is a 2nd-level 1D algorithm.

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

Comparison between two Ps is of variable-depth P hierarchy, with sign at the top, until max higher-layer miss.
This is vertical induction: results of higher-layer comparison predict results of next-layer comparison,
similar to lateral induction: variable-range comparison among Ps, until first match or max prior-Ps miss.

Resulting PPs will be more like 1D graphs, with explicit distances between nearest element Ps.
This is different from 1st level connectivity clustering, where all distances between nearest elements = 1.
'''

ave_dI = 1000                                                                # higher value = more sign change in a line, and lesser positive sign dert_P (observe 13_smP.jpg)
ave_div = 50
ave_rM = .5   # average relative match per input magnitude, at rl=1 or .5?  # lower value = more sign change in a line, and lesser positive sign dert_P(observe 13_smP.jpg)
ave_net_M = 100  # search stop                                              # higher value = more positive sign dert_P (observe 13_smP.jpg)
# no ave_mP: deviation computed via rM  # ave_mP = ave*3: comp cost, or n vars per P: rep cost?


def comp_P(P_):
    dert_P_ = []  # array of alternating-sign Ps with derivatives from comp_P
    comb_dert_P_layers = []

    for i, P in enumerate(P_):
        sign, L, I, D, M, dert_, sub_H, _smP = P  # _smP = 0 in line_patterns, M: deviation even if min
        neg_M = vmP = smP = neg_L = 0  # initialization

        for j, _P in enumerate(P_[i+1 :]):  # no last-P displacement, just shifting first _P for variable-range comp
            _sign, _L, _I, _D, _M, _dert_, _sub_H, __smP = _P

            if M - neg_M > ave_net_M:  # search continues while net_M > ave, True for 1st _P, no selection by M sign
                dL = L - _L
                mL = min(L, _L)  # - ave_rM * L?  L: positions / sign, derived: magnitude-proportional value
                dI = I - _I      # proportional to distance, not I?
                mI = ave_dI - abs(dI)  # I is not derived, match is inverse deviation of miss
                dD = D - _D      # sum if opposite-sign
                mD = min(D, _D)  # - ave_rM * D?  same-sign D in dP?
                dM = M - _M      # sum if opposite-sign
                mM = min(M, _M)  # - ave_rM * M?  negative if x-sign, M += adj_M + deep_M: P value before layer value?

                mP = mL + mM + mD  # match(P, _P) for derived vars, mI is already a deviation
                proj_mP = (L + M + D) * (ave_rM ** (1 + neg_L/L))  # projected mP at current relative distance
                vmP = mI + (mP - proj_mP)  # deviation from projected mP, ~ I*rM contrast value, +|-? replaces mP?
                smP = vmP > 0

                if smP:  # dert_P sign is positive if forward match is found, else negative
                    P_[i+1+j][-1] = True  # backward match per P: __smP = True
                    
                    # next layer dert_P_ computation
                    
                    sub_layer_ = P[6]
                    
                    for sub_layer in sub_layer_:
                        
                        sub_dert_P_layers = [] # not sure should initialize here or before the for loop in line 73, need test further
                        
                        sub_layer_P_ = sub_layer[0][5] # get P_
                        
                        sub_dert_P_layers.append(comp_P(sub_layer_P_))
                    
                        comb_dert_P_layers = [comb_dert_P_layers + sub_dert_P_layers for comb_dert_P_layers, sub_dert_P_layers in
                                   zip_longest(comb_dert_P_layers, sub_dert_P_layers, fillvalue=[])] 
                    
                    # add deeper comp
                    dert_P_.append( (smP, vmP, neg_M, neg_L, mL, dL, mI, dI, mD, dD, mM, dM, P, comb_dert_P_layers)) # should we pack deep_dert_P_ in dert_P_, or somewhere else?

                    break  # nearest-neighbour search, terminated by first match
                else:
                    # accumulate negative-mP params:
                    neg_M += mP  # accumulate contiguous miss: negative mP  # roD / dPP?
                    neg_L += _L  # distance to match, if any
                       
                    # if conditions in line 62 and 78 not met and it is the end of the line, add empty dert_P_ for this input P
                    # this is to prevent lossing derts info in P
                    if _P is P_[-1]: 
                        dert_P_.append((smP or _smP, vmP, neg_M, neg_L, 0, 0, 0, 0, 0, 0, 0, 0, P, comb_dert_P_layers))
                    '''                     
                    no contrast value in neg dert_Ps and PPs: initial opposite-sign P miss is expected
                    neg dert_P derivatives are not significant;  actual decay = neg_M obviates distance * decay '''
            else:
                dert_P_.append((smP or _smP, vmP, neg_M, neg_L, 0, 0, 0, 0, 0, 0, 0, 0, P, comb_dert_P_layers))  # ignore 0s if ~smP
                # smP is ORed bilaterally, negative for single (weak) dert_Ps only
                break  # neg net_M: stop search

    return dert_P_


def form_PPm(dert_P_):  # cluster dert_Ps by mP sign, positive only: no contrast in overlapping comp?
    PPm_ = []
    # initialize PPm with first dert_P:
    _smP, MP, Neg_M, Neg_L, ML, DL, MI, DI, MD, DD, MM, DM, _P, _comb_dert_P_layers = dert_P_[0]  # positive only, no contrast?
    P_ = [_P]
    comb_dert_P_layers_ = [_comb_dert_P_layers]

    for i, dert_P in enumerate(dert_P_, start=1):
        smP = dert_P[0]
        if smP != _smP:
            PPm_.append([_smP, MP, Neg_M, Neg_L, ML, DL, MI, DI, MD, DD, MM, DM, P_, comb_dert_P_layers_])
            # initialize PPm with current dert_P:
            _smP, MP, Neg_M, Neg_L, ML, DL, MI, DI, MD, DD, MM, DM, _P, _comb_dert_P_layers  = dert_P
            P_ = [_P]
            comb_dert_P_layers_ = [_comb_dert_P_layers]
        else:
            # accumulate PPm with current dert_P:
            smP, mP, neg_M, neg_L, mL, dL, mI, dI, mD, dD, mM, dM, P, comb_dert_P_layers = dert_P
            MP+=mP; Neg_M+=neg_M; Neg_L+=neg_L; ML+=mL; DL+=dL; MI+=mI; DI+=dI; MD+=mD; DD+=dD; MM+=mM; DM+=dM
            P_.append(P)
            comb_dert_P_layers_.append(comb_dert_P_layers)
        _smP = smP

    PPm_.append([_smP, MP, Neg_M, Neg_L, ML, DL, MI, DI, MD, DD, MM, DM, P_, comb_dert_P_layers_])  # pack last PP

    # in form_PPd:
    # dP = dL + dM + dD  # -> directional PPd, equal-weight params, no rdn?
    # ds = 1 if Pd > 0 else 0

    ''' evaluation for comp by division is per PP, not per P: results must be comparable between consecutive Ps  

        fdiv = 0, nvars = []
        if M + abs(dL + dI + dD + dM) > ave_div:  
            
            rL = L / _L  # DIV comp L, SUB comp (summed param * rL) -> scale-independent d, neg if cross-sign:
            norm_D = D * rL
            norm_dD = norm_D - _norm_D
            nmDx = min(nDx, _Dx)  # vs. nI = dI * rL or aI = I / L?
            nDy = Dy * rL;
            ndDy = nDy - _Dy;
            nmDy = min(nDy, _Dy)

            Pnm = mX + nmDx + nmDy  # defines norm_mPP, no ndx: single, but nmx is summed

            if Pm > Pnm: nmPP_rdn = 1; mPP_rdn = 0  # added to rdn, or diff alt, olp, div rdn?
            else:        mPP_rdn = 1; nmPP_rdn = 0
            Pnd = ddX + ndDx + ndDy  # normalized d defines norm_dPP or ndPP

            if Pd > Pnd: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
            else:        dPP_rdn = 1; ndPP_rdn = 0
            fdiv = 1
            nvars = Pnm, nmD, mPP_rdn, nmPP_rdn, Pnd, ndDx, ndDy, dPP_rdn, ndPP_rdn

        else:
            fdiv = 0  # DIV comp flag
            nvars = 0  # DIV + norm derivatives
        '''
    return PPm_
