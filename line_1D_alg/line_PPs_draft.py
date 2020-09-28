'''
line_PPs is a 2nd-level 1D algorithm, processing output Ps from the 1st level: line_patterns.
It cross-compares Ps (s, L, I, D, M, dert_, layers) and evaluates them for deeper cross-comparison.

Range or derivation of cross-comp is selectively increased if the match from prior-order cross-comp is above threshold:
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
import numpy as np
from line_patterns_defs import Cdert_P, CPP

ave_dI = 1000
ave_div = 50
ave_rM = .5  # average relative match per input magnitude, at rl=1 or .5?
ave_net_M = 100  # search stop
ave_Ls = 3

# no ave_mP: deviation computed via rM  # ave_mP = ave*3: comp cost, or n vars per P: rep cost?

def comp_P_(P_):  # cross-compare patterns within horizontal line
    dert_P_ = []  # comp_P_ forms array of alternating-sign dert_Ps (derivatives + P): output of pair-wise comp_P

    for i, P in enumerate(P_):
        neg_M = vmP = smP = _smP = neg_L = 0  # initialization
        M = P.M
        for j, _P in enumerate(P_[i + 1:]):  # variable-range comp, no last-P displacement, just shifting first _P
            if M - neg_M > ave_net_M:
                # search while net_M > ave, True for 1st _P, no select by M sign
                dert_P, _L, _smP = comp_P(P, _P, neg_M, neg_L)
                smP, vmP, neg_M, neg_L, P = dert_P.smP, dert_P.MP, dert_P.Neg_M, dert_P.Neg_L, dert_P.P
                if smP:
                    P_[i + 1 + j].smP = True  # backward match per P: __smP = True
                    dert_P_.append(dert_P)
                    break  # nearest-neighbour search is terminated by first match
                else:
                    neg_M += vmP  # accumulate contiguous miss: negative mP
                    neg_L += _L  # accumulate distance to match
                    if j == len(P_):  # last P
                        dert_P_.append(Cdert_P(smP=smP or _smP, MP=vmP, Neg_M=neg_M, Neg_L=neg_L, P=P ))  # unconnected P, derivatives are ignored
                    '''                     
                    no contrast value in neg dert_Ps and PPs: initial opposite-sign P miss is expected
                    neg_dert_P derivatives are not significant; neg_M obviates distance * decay_rate * M '''
            else:
                dert_P_.append(Cdert_P(smP=smP or _smP, MP=vmP, Neg_M=neg_M, Neg_L=neg_L, P=P))
                # smP is ORed bilaterally, negative for single (weak) dert_Ps only
                break  # neg net_M: stop search

    return dert_P_


def comp_P(P, _P, neg_M, neg_L):
    # _smP = 0 in line_patterns, M: deviation even if min
    sign, L, I, D, M, dert_, sub_H, _smP = P.sign, P.L, P.I, P.D, P.M, P.dert_, P.sub_layers, P.smP
    _sign, _L, _I, _D, _M, _dert_, _sub_H, __smP = _P.sign, _P.L, _P.I, _P.D, _P.M, _P.dert_, _P.sub_layers, _P.smP

    dL = L - _L
    mL = min(L, _L)  # - ave_rM * L?  L: positions / sign, derived: magnitude-proportional value
    dI = I - _I  # proportional to distance, not I?
    mI = ave_dI - abs(dI)  # I is not derived, match is inverse deviation of miss
    dD = D - _D  # sum if opposite-sign
    mD = min(D, _D)  # - ave_rM * D?  same-sign D in dP?
    dM = M - _M  # sum if opposite-sign
    mM = min(M, _M)  # - ave_rM * M?  negative if x-sign, M += adj_M + deep_M: P value before layer value?

    mP = mL + mM + mD  # match(P, _P) for derived vars, mI is already a deviation
    proj_mP = (L + M + D) * (ave_rM ** (1 + neg_L / L))  # projected mP at current relative distance
    vmP = mI + (mP - proj_mP)  # deviation from projected mP, ~ I*rM contrast value, +|-? replaces mP?
    smP = vmP > 0

    if smP:  # forward match, compare sub_layers between P.sub_H and _P.sub_H (sub_hierarchies):
        dert_sub_H = []
        if P.sub_layers and _P.sub_layers:  # not empty sub layers
            for sub_P, _sub_P in zip(P.sub_layers, _P.sub_layers):
                if P and _P:  # both forks exist
                    Ls, fdP, fid, rdn, rng, sub_P_ = sub_P[0]
                    _Ls, _fdP, _fid, _rdn, _rng, _sub_P_ = _sub_P[0]
                    # fork comparison:
                    if fdP == _fdP and rng == _rng and min(Ls, _Ls) > ave_Ls:
                        dert_sub_P_ = []
                        sub_MP = 0
                        # compare all sub_Ps to each _sub_P, form dert_sub_P per compared pair
                        for sub_P in sub_P_:
                            for _sub_P in _sub_P_:
                                dert_sub_P, _, _ = comp_P(sub_P, _sub_P, neg_M=0, neg_L=0)  # ignore _sub_L, _sub_smP?
                                sub_MP += dert_sub_P.MP  # sum sub_vmPs in dert_P_layer
                                dert_sub_P_.append(dert_sub_P)

                        dert_sub_H.append((fdP, fid, rdn, rng, dert_sub_P_))  # only layers that have been compared
                        vmP += sub_MP  # of compared H, no specific mP?
                        if sub_MP < ave_net_M:
                            # or mH: trans-layer induction?
                            break  # low vertical induction, deeper sub_layers are not compared
                    else:
                        break  # deeper P and _P sub_layers are from different intra_comp forks, not comparable?

    dert_P = Cdert_P(smP=smP, MP=vmP, Neg_M=neg_M, Neg_L=neg_L, P=P, ML=mL, DL=dL, MI=mI, DI=dI, MD=mD, DD=dD, MM=mM, DM=dM)
    return dert_P, _L, _smP


def form_PPm(dert_P_):  # cluster dert_Ps by mP sign, positive only: no contrast in overlapping comp?
    PPm_ = []
    # initialize PPm with first dert_P:
    dert_P = dert_P_[0]
     # positive only, no contrast?
    _smP, MP, Neg_M, Neg_L, _P, ML, DL, MI, DI, MD, DD, MM, DM = \
        dert_P.smP, dert_P.MP, dert_P.Neg_M, dert_P.Neg_L, dert_P.P, \
        dert_P.ML, dert_P.DL, dert_P.MI, dert_P.DI, dert_P.MD, dert_P.DD, dert_P.MM, dert_P.DM
    P_ = [_P]

    for i, dert_P in enumerate(dert_P_, start=1):
        smP = dert_P.smP
        if smP != _smP:
            PPm_.append(CPP(smP=smP, MP=MP, Neg_M=Neg_M, Neg_L=Neg_L, P_=P_, ML=ML, DL=DL,MI=MI, DI=DI, MD=MD, DD=DD, MM=MM, DM=DM))
            # initialize PPm with current dert_P:
            _smP, MP, Neg_M, Neg_L, _P, ML, DL, MI, DI, MD, DD, MM, DM = \
                dert_P.smP, dert_P.MP, dert_P.Neg_M, dert_P.Neg_L, dert_P.P, \
                dert_P.ML, dert_P.DL, dert_P.MI, dert_P.DI, dert_P.MD, dert_P.DD, dert_P.MM, dert_P.DM
            P_ = [_P]
        else:
            # accumulate PPm with current dert_P:
            smP, mP, neg_M, neg_L, P, mL, dL, mI, dI, mD, dD, mM, dM = \
                dert_P.smP, dert_P.MP, dert_P.Neg_M, dert_P.Neg_L, dert_P.P, \
                dert_P.ML, dert_P.DL, dert_P.MI, dert_P.DI, dert_P.MD, dert_P.DD, dert_P.MM, dert_P.DM
            MP+=mP; Neg_M+=neg_M; Neg_L+=neg_L; ML+=mL; DL+=dL; MI+=mI; DI+=dI; MD+=mD; DD+=dD; MM+=mM; DM+=dM

            P_.append(P)
        _smP = smP
    # pack last PP
    PPm_.append(CPP(smP=_smP, MP=MP, Neg_M=Neg_M, Neg_L=Neg_L, P_=P_, ML=ML, DL=DL,MI=MI, DI=DI, MD=MD, DD=DD, MM=MM, DM=DM))

    return PPm_

''' Each PP is evaluated for intra-processing: 
    incremental range and derivation as in line_patterns intra_P, 
    div_comp if L-proportional contents: comp norm param or x param ratios, diffs are not L-proportional? 
    form_par_P if param Match | x_param Contrast: diff (D_param, ave_D_alt_params: co-derived co-vary? neg val per P, else delete?

    in form_PPd:
    dP = dL + dM + dD  # -> directional PPd, equal-weight params, no rdn?
    ds = 1 if Pd > 0 else 0
'''

def div_comp_P(PP_):  # draft, check all PPs for div_comp among their element Ps
    '''
    evaluation for comp by division is per PP, not per P: results must be comparable between consecutive Ps
    estimated value of division = rm * D, compression = D - min * N: vertical vs lateral. if same-sign:

    * higher-derivation D: ratio induction?
    * higher-composition D: vertical d-induction from L, div value?
    sum comp -> rVar | ave comp: Var*rL -> dVar?
    '''
    for PP in PP_:
        if PP.M / (PP.L + PP.I + abs(PP.D) + abs(PP.dM)) * (abs(PP.dL) + abs(PP.dI) + abs(PP.dD) + abs(PP.dM)) > ave_div:
            # if irM * D_vars: match rate projects der and div match,
            # div if scale invariance: comp x dVars, signed
            ''' 
            | abs(dL + dI + dD + dM): div value ~= L, Vars correlation: stability of density, opposite signs cancel-out?
            | div_comp value is match: min(dL, dI, dD, dM) * 4, | sum of pairwise mins?
            '''
            _dert_P = PP.dert_P_[0]
            # smP, vmP, neg_M, neg_L, iP, mL, dL, mI, dI, mD, dD, mM, dM = P,
            _sign, _L, _I, _D, _M, _dert_, _sub_H, __smP = _dert_P[4]

            for i, dert_P in enumerate(PP.dert_P_[1:]):
                sign, L, I, D, M, dert_, sub_H, _smP = dert_P[4]
                # DIV comp L, SUB comp (summed param * rL) -> scale-independent d, neg if cross-sign:
                rL = L / _L
                # mL = whole_rL * min_L?
                dI = I * rL - _I  # rL-normalized dI, vs. nI = dI * rL or aI = I / L
                mI = ave_dI - abs(dI)  # I is not derived, match is inverse deviation of miss
                dD = D * rL - _D  # sum if opposite-sign
                mD = min(D, _D)   # same-sign D in dP?
                dM = M * rL - _M  # sum if opposite-sign
                mM = min(M, _M)   # - ave_rM * M?  negative if x-sign, M += adj_M + deep_M: P value before layer value?

                mP = mI + mM + mD  # match(P, _P) for derived vars, mI is already a deviation
                                   # defines norm_mPP, no ndx: single, but nmx is summed
                if mP > dert_P[1]:
                    rrdn = 1  # added to rdn, or diff alt, olp, div rdn?
                else:
                    rrdn = 2
                if mP > ave_dI * 3 * rrdn:
                    rvars = mP, mI, mD, mM, dI, dD, dM  # dPP_rdn, ndPP_rdn
                else:
                    rvars = []
                # append rrdn and ratio variables to current dert_P:
                PP.dert_P_[i] += [rrdn, rvars]
                # P vars -> _P vars:
                _sign = sign, _L = L, _I = I, _D = D, _M = M, _dert_ = dert_, _sub_H = sub_H, __smP = _smP
                '''
                also define dP,
                if Pd > Pnd: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
                else:        dPP_rdn = 1; ndPP_rdn = 0
                '''
    return PP_


''' non-class versions: '''
'''
def accum_PP(PP: dict, **params) -> None:
    PP.update({param: PP[param] + value for param, value in params.items()})


def comp_P_(P_):  # cross-compare patterns within horizontal line
    dert_P_ = []  # comp_P_ forms array of alternating-sign dert_Ps (derivatives + P): output of pair-wise comp_P

    for i, P in enumerate(P_):
        neg_M = vmP = smP = _smP = neg_L = 0  # initialization
        M = P[4]
        for j, _P in enumerate(P_[i+1 :]):  # variable-range comp, no last-P displacement, just shifting first _P
            if M - neg_M > ave_net_M:
                # search while net_M > ave, True for 1st _P, no select by M sign
                dert_P, _L, _smP = comp_P(P, _P, neg_M, neg_L)
                smP, vmP, neg_M, neg_L, P = dert_P[:5]
                if smP:
                    P_[i + 1 + j][-1] = True  # backward match per P: __smP = True
                    dert_P_.append(dert_P)
                    break  # nearest-neighbour search is terminated by first match
                else:
                    neg_M += vmP  # accumulate contiguous miss: negative mP
                    neg_L += _L   # accumulate distance to match
                    if j == len(P_):  # last P
                        dert_P_.append((smP or _smP, vmP, neg_M, neg_L, P, 0, 0, 0, 0, 0, 0, 0, 0))          
                    # no contrast value in neg dert_Ps and PPs: initial opposite-sign P miss is expected
                    # neg_dert_P derivatives are not significant; neg_M obviates distance * decay_rate * M 
            else:
                dert_P_.append((smP or _smP, vmP, neg_M, neg_L, P, 0, 0, 0, 0, 0, 0, 0, 0))
                # smP is ORed bilaterally, negative for single (weak) dert_Ps
                break  # neg net_M: stop search

    return dert_P_


def comp_P(P, _P, neg_M, neg_L):

    sign, L, I, D, M, dert_, sub_H, _smP = P  # _smP = 0 in line_patterns, M: deviation even if min
    _sign, _L, _I, _D, _M, _dert_, _sub_H, __smP = _P

    dL = L - _L
    mL = min(L, _L)  # - ave_rM * L?  L: positions / sign, derived: magnitude-proportional value
    dI = I - _I  # proportional to distance, not I?
    mI = ave_dI - abs(dI)  # I is not derived, match is inverse deviation of miss
    dD = D - _D  # sum if opposite-sign
    mD = min(D, _D)  # - ave_rM * D?  same-sign D in dP?
    dM = M - _M  # sum if opposite-sign
    mM = min(M, _M)  # - ave_rM * M?  negative if x-sign, M += adj_M + deep_M: P value before layer value?

    mP = mL + mM + mD  # match(P, _P) for derived vars, mI is already a deviation
    proj_mP = (L + M + D) * (ave_rM ** (1 + neg_L / L))  # projected mP at current relative distance
    vmP = mI + (mP - proj_mP)  # deviation from projected mP, ~ I*rM contrast value, +|-? replaces mP?
    smP = vmP > 0

    if smP:  # forward match, compare sub_layers between P.sub_H and _P.sub_H (sub_hierarchies):
        dert_sub_H = []
        if P[6] and _P[6]: # not empty sub layers
            for (Ls, fdP, fid, rdn, rng, sub_P_), (_Ls, _fdP, _fid, _rdn, _rng, _sub_P_) in zip(*P[6], *_P[6]):
                # fork comparison:
                if fdP == _fdP and rng == _rng and min(Ls, _Ls) > ave_Ls:
                    dert_sub_P_ = []
                    sub_MP = 0
                    # compare all sub_Ps to each _sub_P, form dert_sub_P per compared pair
                    for sub_P in sub_P_:
                        for _sub_P in _sub_P_:
                            dert_sub_P, _, _ = comp_P(sub_P, _sub_P, neg_M=0, neg_L=0)  # ignore _sub_L, _sub_smP?
                            sub_MP += dert_sub_P[1]  # sum sub_vmPs in dert_P_layer
                            dert_sub_P_.append(dert_sub_P)

                    dert_sub_H.append((fdP, fid, rdn, rng, dert_sub_P_))  # only layers that have been compared
                    vmP += sub_MP  # of compared H, no specific mP?
                    if sub_MP < ave_net_M:
                        # or mH: trans-layer induction?
                        break  # low vertical induction, deeper sub_layers are not compared
                else:
                    break  # deeper P and _P sub_layers are from different intra_comp forks, not comparable?

    return (smP, vmP, neg_M, neg_L, P, mL, dL, mI, dI, mD, dD, mM, dM), _L, _smP


def form_PPm(dert_P_):  # cluster dert_Ps by mP sign, positive only: no contrast in overlapping comp?
    PPm_ = []
    # initialize PPm with first dert_P:
    _smP, MP, Neg_M, Neg_L, _P, ML, DL, MI, DI, MD, DD, MM, DM = dert_P_[0]  # positive only, no contrast?
    P_ = [_P]

    for i, dert_P in enumerate(dert_P_, start=1):
        smP = dert_P[0]
        if smP != _smP:
            PPm_.append([_smP, MP, Neg_M, Neg_L, P_, ML, DL, MI, DI, MD, DD, MM, DM])
            # initialize PPm with current dert_P:
            _smP, MP, Neg_M, Neg_L, _P, ML, DL, MI, DI, MD, DD, MM, DM = dert_P
            P_ = [_P]
        else:
            # accumulate PPm with current dert_P:
            smP, mP, neg_M, neg_L, P, mL, dL, mI, dI, mD, dD, mM, dM = dert_P
            MP+=mP; Neg_M+=neg_M; Neg_L+=neg_L; ML+=mL; DL+=dL; MI+=mI; DI+=dI; MD+=mD; DD+=dD; MM+=mM; DM+=dM

            P_.append(P)
        _smP = smP

    PPm_.append([_smP, MP, Neg_M, Neg_L, P_, ML, DL, MI, DI, MD, DD, MM, DM])  # pack last PP

    return PPm_
    # in form_PPd:
    # dP = dL + dM + dD  # -> directional PPd, equal-weight params, no rdn?
    # ds = 1 if Pd > 0 else 0

def accum_PP(PP: dict, **params) -> None:
    PP.update({param: PP[param] + value for param, value in params.items()})

'''