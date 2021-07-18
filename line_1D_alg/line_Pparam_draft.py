'''
line_Pps is a 2nd-level 1D algorithm, its input is Ps formed by the 1st-level line_patterns.
It cross-compares P params (initially L, I, D, M) and forms Pparam_ for each of them
'''

# add ColAlg folder to system path
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname("CogAlg"), '..')))

from line_patterns import CP
from frame_2D_alg.class_cluster import ClusterStructure, comp_param

class CderP(ClusterStructure):  # not used
    sign = bool
    rrdn =int
    neg_M = int
    neg_L = int
    adj_mP = int
    P = object
    layer1 = dict  # dert per compared param
    der_sub_H = list  # sub hierarchy of derivatives, from comp sublayers

class Cpdert(ClusterStructure):
    i = int  # input for range_comp only
    p = int  # accumulated in rng
    d = int  # accumulated in rng
    m = int  # distinct in deriv_comp only
    negL = int  # in mdert only
    negM = int  # in mdert only

class CPp(CP):

    sign = bool
    dert_ = list # contains i, p, d, m
    sublayers = list
    derPp = object  # forward comp_P derivatives

    fPd = bool  # P is Pd if true, else Pm; also defined per layer
    negL = int  # in mdert only
    negM = int  # in mdert only

ave = 100  # ave dI -> mI, * coef / var type
# no ave_mP: deviation computed via rM  # ave_mP = ave* n_comp_params: comp cost, or n vars per P: rep cost?
ave_div = 50
ave_rM = .5  # average relative match per input magnitude, at rl=1 or .5?
ave_M = 100  # search stop
ave_D = 100
ave_sub_M = 500  # sub_H comp filter
ave_Ls = 3
ave_PPM = 200
ave_merge = 50  # merge adjacent Ps
ave_inv = 20 # ave inverse m, change to Ave from the root intra_blob?
ave_min = 5  # ave direct m, change to Ave_min from the root intra_blob?


def search(P_):  # cross-compare patterns within horizontal line

    sub_search_recursive(P_)  # search with incremental distance: first inside sublayers
    merge_P_draft(P_, merge_eval=merge_eval_P, fPd=0)  # merge I- or D- similar and weakly separated Ps

    layer0 = {'L_': [[],.25], 'I_': [[],.5], 'D_': [[],.25], 'M_': [[],.5]}  # M is doubled because it represents both comparands
    if len(P_) > 1:  # at least 2 comparands, unpack P_:
        for P in P_:
            layer0['L_'][0].append((P.L, P.L, P.x0))  # L: (2 L entry for code consistent processing later)
            layer0['I_'][0].append((P.I, P.L, P.x0))  # I
            layer0['D_'][0].append((P.D, P.L, P.x0))  # D
            layer0['M_'][0].append((P.M, P.L, P.x0))  # M

        for n, param_name in enumerate(layer0):  # loop L_, I_, D_, M_
            search_param_(param_name, layer0[param_name])

    return layer0

def sub_search_recursive(P_):  # search in top sublayer per P / sub_P

    for P in P_:
        if P.sublayers:
            sublayer = P.sublayers[0][0]  # top sublayer has one element
            sub_P_ = sublayer[5]
            if len(sub_P_) > 2:
                PM = P.M; PD = P.D
                # include match added by last search (not sure with Pp)
                if P.fPd:
                    if abs(PD) > ave_D:  # better use sublayers.D|M, but we don't have it yet
                        sublayer[6].append(search(sub_P_))
                        sub_search_recursive(sub_P_)  # deeper sublayers search is selective per sub_P
                elif PM > ave_M:
                    sublayer[6].append(search(sub_P_))
                    sub_search_recursive(sub_P_)  # deeper sublayers search is selective per sub_P


def search_param_(param_name, iparam):

    ddert_, mdert_ = [], []  # line-wide (i, p, d, m)_, + (negL, negM) in mdert: variable-range search
    rdn = iparam[1]
    param_ = iparam[0]
    _param, _L, _x0 = param_[0]

    for i, (param, L, x0) in enumerate(param_[1:], start=1):
        dert = comp_param(_param, param, param_name, ave/rdn)  # param is compared to prior-P param
        # or ave_merge * dist_coef: ave_rM ** (1 + negL / (negL+_L))  # average match projected at current distance?

        ddert_.append( Cpdert( i=dert.i, p=dert.p, d=dert.d, m=dert.m))  # negL, negM stay 0
        negL = negM = 0
        comb_M = dert.m
        j = i
        while comb_M > 0 and j+1 < len(param_):
            j += 1
            ext_param, ext_L, ext_x0 = param_[j]  # extend search beyond next param
            dert = comp_param(param, ext_param, param_name, ave)
            comb_M = dert.m + negM - ave / (1 - (negL / (negL + L + _L)))
            negM += dert.m
            negL += ext_L
        # after extended search, if any:
        mdert_.append( Cpdert( i=dert.i, p=dert.p, d=dert.d, m=dert.m, negL=negL, negM=negM))

        _param = param

    Ppm_ = form_Pp_(mdert_, fPd=0)
    Ppd_ = form_Pp_(ddert_, fPd=1)

    iparam[0] = (Ppm_, Ppd_)


def form_Pp_(dert_, fPd):  # almost the same as form_P_ for now
    # initialization:
    Pp_ = []
    x = 0
    _sign = None  # to initialize 1st P, (None != True) and (None != False) are both True

    for dert in dert_:  # segment by sign
        if fPd: sign = dert.d > 0
        else:   sign = dert.m > 0

        if sign != _sign:
            # sign change, initialize P and append it to P_
            Pp = CPp(sign=_sign, L=1, I=dert.p, D=dert.d, M=dert.m, negL=dert.negL, negM=dert.negM, x0=0, dert_=[dert], sublayers=[], fPd=fPd)
            Pp_.append(Pp)  # updated with accumulation below
        else:
            # accumulate params:
            Pp.L += 1; Pp.I += dert.p; Pp.D += dert.d; Pp.M += Pp.dert.m; Pp.negL+=dert.negL; Pp.negM+=dert.negM
            Pp.dert_ += [dert]
        x += 1
        _sign = sign

    Pp_.append(Pp)  # last incomplete Pp
    merge_P_draft(Pp_, merge_eval=merge_eval_Pp, fPd=fPd)  # add eval by negL and negM

    return Pp_


def merge_P_draft(P_, merge_eval, fPd):
    '''
    Initial P separation is determined by pixel-level sign change, but resulting opposite-sign pattern may be relatively weak,
    and same-sign patterns it separates relatively strong.
    Another criterion to re-evaluate separation is similarity of defining param: M/L for Pm, D/L for Pd, among the three Ps
    If relative proximity * relative similarity > ave? all three Ps should be merged into one.
    '''
    new_P_ = []
    while len(P_) > 2:  # at least 3 Ps
        __P = P_.pop(0)
        _P = P_.pop(0)
        P = P_.pop(0)

        if merge_eval(__P, _P, P, fPd) > ave_merge:
            # or ave_merge * dist_coef: ave_rM ** (1 + _P.L / merged_P.L)  # average match projected at current distance between P and __P?
            # for debugging
            print('P_'+str(_P.id)+' and P_'+str(P.id)+' are merged into P_'+str(__P.id))
            # merge _P and P into __P
            for merge_P in [_P, P]:
                __P.x0 = min(__P.x0, merge_P.x0)
                __P.accum_from(merge_P)
                __P.dert_+= merge_P.dert_

            __P = merge_P_back(new_P_, __P, merge_eval, fPd)  # backward merging
            P_.insert(0, __P)  # insert merged __P back into P_ to continue merging
        else:
            new_P_.append(__P) # append __P to P_ when there is no further merging process for __P
            P_.insert(0, P)    # insert P back into P_ for the consecutive merging process
            P_.insert(0, _P)  # insert _P back into P_ for the consecutive merging process

    # pack remaining Ps:
    if P_: new_P_ += P_

    return new_P_


def merge_P_back(new_P_, P, merge_eval, fPd):  # P is __P in calling merge_P_draft

    while len(new_P_) > 2:  # at least 3 Ps
        _P = new_P_.pop()
        __P = new_P_.pop()

        if merge_eval(__P, _P, P, fPd) > ave_merge:
            # or ave_merge * dist_coef: ave_rM ** (1 + _P.L / merged_P.L)  # average match projected at current distance between P and __P?
            # for debug purpose
            print('P_'+str(_P.id)+' and P_'+str(P.id)+' are backward merged into P_'+str(__P.id))
            # merge _P and P into __P
            for merge_P in [_P, P]:
                __P.x0 = min(__P.x0, merge_P.x0)
                __P.accum_from(merge_P)
                __P.dert_+= merge_P.dert_
            P = __P  # also returned
        else:
            new_P_+= [__P, _P]
            break

    return P

def merge_eval_P(__P, _P, P, fPd):
    '''
    For 3 Pms, same-sign P1 and P3, opposite-sign P2:
    relative proximity = abs((M1+M3) / M2)
    relative similarity = match (M1/L1, M3/L3) / miss (match (M1/L1, M2/L2) + match (M3/L3, M2/L2)) # both should be negative
    '''
    if fPd:
        proximity = abs((__P.D + P.D) / _P.D) if _P.D != 0 else 0  # prevents /0
        __mean=__P.D/__P.L; _mean=_P.D/_P.L; mean=P.D/P.L
    else:
        proximity = abs((__P.M + P.M) / _P.M) if _P.M != 0 else 0  # prevents /0
        __mean=__P.M/__P.L; _mean=_P.M/_P.L; mean=P.M/P.L # typo
    m13 = min(mean, __mean) - abs(mean-__mean)/2   # P1 & P3
    m12 = min(_mean, __mean) - abs(_mean-__mean)/2 # P1 & P2
    m23 = min(_mean, mean) - abs(_mean- mean)/2    # P2 & P3

    similarity = m13 / abs( m12 + m23)  # both should be negative
    merge_value = proximity * similarity

    return merge_value


def merge_eval_Pp(__Pp, _Pp, Pp, fPd):
    '''
    For 3 Pps, same-sign Pp1 and Pp3, opposite-sign Pp2:
    relative proximity = abs((M1+M3) / M2)
    relative similarity = match (M1/L1, M3/L3) / miss (match (M1/L1, M2/L2) + match (M3/L3, M2/L2)) # both should be negative
    '''
    if fPd:
        proximity = abs((__Pp.D + Pp.D) / _Pp.D) if _Pp.D != 0 else 0  # prevents /0
        __mean=__Pp.D/__Pp.L
        _mean=_Pp.D/_Pp.L
        mean=Pp.D/Pp.L
    else:
        proximity = abs((__Pp.M-__Pp.negM + Pp.M-Pp.negM) / (_Pp.M-_Pp.negM) ) if (_Pp.M-_Pp.negM) != 0 else 0  # prevents /0
        __mean= (__Pp.M-__Pp.negM)/(__Pp.L+__Pp.negL)
        _mean= (_Pp.M-_Pp.negM)/(_Pp.L+_Pp.negL)
        mean= (Pp.M-Pp.negM)/(Pp.L, Pp.negL)
    m13 = min(mean, __mean) - abs(mean-__mean)/2   # P1 & P3
    m12 = min(_mean, __mean) - abs(_mean-__mean)/2 # P1 & P2
    m23 = min(_mean, mean) - abs(_mean- mean)/2    # P2 & P3

    similarity = m13 / abs( m12 + m23)  # both should be negative
    merge_value = proximity * similarity

    return merge_value

# below is not revised
'''
def sub_search_recursive(P_, fderP):  # search in top sublayer per P / sub_P

    for P in P_:
        if P.sublayers:
            sublayer = P.sublayers[0][0]  # top sublayer has one element
            sub_P_ = sublayer[5]
            if len(sub_P_) > 2:
                PM = P.M; PD = P.D
                if fderP:
                    PM += P.derP.mP; PD += P.derP.mP
                    # include match added by last search
                if P.fPd:
                    if abs(PD) > ave_D:  # better use sublayers.D|M, but we don't have it yet
                        sub_PPm_, sub_PPd_ = search(sub_P_)
                        sublayer[6].append(sub_PPm_); sublayer[7].append(sub_PPd_)
                        sub_search_recursive(sub_P_, fderP=1)  # deeper sublayers search is selective per sub_P
                elif PM > ave_M:
                    sub_PPm_, sub_PPd_ = search(sub_P_)
                    sublayer[6].append(sub_PPm_); sublayer[7].append(sub_PPd_)
                    sub_search_recursive(sub_P_, fderP=1)  # deeper sublayers search is selective per sub_P

'''

def comp_P(_P, P, neg_L, neg_M):  # multi-variate cross-comp, _smP = 0 in line_patterns

    mP = dP = 0
    layer1 = dict({'L':.0,'I':.0,'D':.0,'M':.0})
    dist_coef = ave_rM ** (1 + neg_L / _P.L)  # average match projected at current distance:

    for param_name in layer1:
        if param_name == "I":
            dist_ave = ave_inv * dist_coef
        else:
            dist_ave = ave_min * dist_coef
        param = getattr(_P, param_name)
        _param = getattr(P, param_name)
        dm = comp_param(_param, param, [], dist_ave)
        rdn = layer0_rdn[param_name]
        mP += dm.m * rdn
        dP += dm.d * rdn
        layer1[param_name] = dm
        '''
        main comp is between summed params, with an option for div_comp, etc.
        mP -= ave_M * ave_rM ** (1 + neg_L / P.L)  # average match projected at current distance: neg_L, add coef / var?
        match(P,_P), ave_M is addition to ave? or abs for projection in search?
        '''
        if P.sign == _P.sign: mP *= 2  # sign is MSB, value of sign match = full magnitude match?
        sign = mP > 0
        if sign:  # positive forward match, compare sublayers between P.sub_H and _P.sub_H:
            comp_sublayers(_P, P, mP)

        if isinstance(_P.derP, CderP):  # derP is created in comp_sublayers
            _P.derP.sign = sign
            _P.derP.layer1 = layer1
            _P.derP.accumulate(mP=mP, neg_M=neg_M, neg_L=neg_L, P=_P)
            derP = _P.derP
        else:
            derP = CderP(sign=sign, mP=mP, neg_M=neg_M, neg_L=neg_L, P=_P, layer1=layer1)
            _P.derP = derP

    return derP, _P.L, _P.sign


def comp_sublayers(_P, P, mP):  # also add dP?

    if P.sublayers and _P.sublayers:  # not empty sub layers
        for _sub_layer, sub_layer in zip(_P.sublayers, P.sublayers):

            if _sub_layer and sub_layer:
                _Ls, _fdP, _fid, _rdn, _rng, _sub_P_, [], [] = _sub_layer[0]
                Ls, fdP, fid, rdn, rng, sub_P_, [], [] = sub_layer[0]
                # fork comparison:
                if fdP == _fdP and rng == _rng and min(Ls, _Ls) > ave_Ls:
                    der_sub_P_ = []
                    sub_mP = 0
                    # compare all sub_Ps to each _sub_P, form dert_sub_P per compared pair:
                    remove_index = []
                    for m, _sub_P in enumerate(_sub_P_):  # note name recycling in nested loop
                        for n, sub_P in enumerate(sub_P_):
                            if n not in remove_index:
                                # -1 for i, because comparing different sub_P_
                                der_sub_P, _, _ = merge_comp_P(_sub_P_, _sub_P, sub_P, -1, n, 0, 0, remove_index)
                                sub_mP += der_sub_P.mP  # sum sub_vmPs in derP_layer
                                der_sub_P_.append(der_sub_P)

                    # delete the merged sub_Ps at last
                    for index in sorted(remove_index, reverse=True):
                        del sub_P_[index]

                    if not isinstance(_P.derP, CderP): _P.derP = CderP(P=_P)  # _P had no derP
                    _P.derP.der_sub_H.append((fdP, fid, rdn, rng, der_sub_P_))  # add only layers that have been compared

                    mP += sub_mP  # of compared H, no specific mP?
                    if sub_mP < ave_sub_M:
                        # potentially mH: trans-layer induction: if mP + sub_mP < ave_sub_M: both local and global values of mP.
                        break  # low vertical induction, deeper sublayers are not compared
                else:
                    break  # deeper P and _P sublayers are from different intra_comp forks, not comparable?

def div_comp_P(PP_):  # draft, check all PPs for x-param comp by division between element Ps
    '''
    div x param if projected div match: compression per PP, no internal range for ind eval.
    ~ (L*D + L*M) * rm: L=min, positive if same-sign L & S, proportional to both but includes fractional miss
    + PPm' DL * DS: xP difference compression, additive to x param (intra) compression: S / L -> comp rS
    also + ML * MS: redundant unless min or converted?
    vs. norm param: Var*rL-> comp norm param, simpler but diffs are not L-proportional?
    '''
    for PP in PP_:
        vdP = (PP.adj_mP + PP.P.M) * abs(PP.dP) - ave_div
        if vdP > 0:
            # if irM * D_vars: match rate projects der and div match,
            # div if scale invariance: comp x dVars, signed
            ''' 
            | abs(dL + dI + dD + dM): div value ~= L, Vars correlation: stability of density, opposite signs cancel-out?
            | div_comp value is match: min(dL, dI, dD, dM) * 4, | sum of pairwise mins?
            '''
            _derP = PP.derP_[0]
            # smP, vmP, neg_M, neg_L, iP, mL, dL, mI, dI, mD, dD, mM, dM = P,
            #_sign, _L, _I, _D, _M, _dert_, _sub_H, __smP = _derP.P
            _P = _derP.P
            for i, derP in enumerate(PP.derP_[1:]):
                P = derP.P
                # DIV comp L, SUB comp (summed param * rL) -> scale-independent d, neg if cross-sign:
                rL = P.L / _P.L
                # mL = whole_rL * min_L?
                '''
                dI = I * rL - _I  # rL-normalized dI, vs. nI = dI * rL or aI = I / L
                mI = ave - abs(dI)  # I is not derived, match is inverse deviation of miss
                dD = D * rL - _D  # sum if opposite-sign
                mD = min(D, _D)   # same-sign D in dP?
                dM = M * rL - _M  # sum if opposite-sign
                mM = min(M, _M)   # - ave_rM * M?  negative if x-sign, M += adj_M + deep_M: P value before layer value?
                mP = mI + mM + mD  # match(P, _P) for derived vars, defines norm_PPm, no ndx: single, but nmx is summed
                '''
                for (param, _param) in zip([P.I, P.D, P.M], [_P.I, _P.D, _P.M]):
                    dm = comp_param(param, _param, [], ave, rL)
                    layer1.append([dm.d, dm.m])
                    mP += dm.m; dP += dm.d

                if dP > P.derP.dP:
                    ndP_rdn = 1; dP_rdn = 0  #Not sure what to do with these
                else:
                    dP_rdn = 1; ndP_rdn = 0

                if mP > derP.mP:
                    rrdn = 1  # added to rdn, or diff alt, olp, div rdn?
                else:
                    rrdn = 2
                if mP > ave * 3 * rrdn:
                    #rvars = mP, mI, mD, mM, dI, dD, dM  # redundant vars: dPP_rdn, ndPP_rdn, assigned in each fork?
                    rvars = layer1
                else:
                    rvars = []
                # append rrdn and ratio variables to current derP:
                #PP.derP_[i] += [rrdn, rvars]
                PP.derP_[i].rrdn = rrdn; PP.derP_[i].layer1 = rvars
                # P vars -> _P vars:
                _P = P
                '''
                m and d from comp_rate is more accurate than comp_norm?
                rm, rd: rate value is relative? 
                also define Pd, if strongly directional? 
                if dP > ndP: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
                else:        dPP_rdn = 1; ndPP_rdn = 0
                '''
    return PP_

def form_adjacent_mP(derP_d_):

    pri_mP = derP_d_[0].mP
    mP = derP_d_[1].mP
    derP_d_[0].adj_mP = derP_d_[1].mP

    for i, derP in enumerate(derP_d_[2:]):
        next_mP = derP.mP
        derP_d_[i+1].adj_mP = (pri_mP + next_mP)/2
        pri_mP = mP
        mP = next_mP

    return derP_d_

def intra_PPm_(PPm_, rdn):
    '''
    Each PP is evaluated for intra-processing, non-recursive here:
    - incremental range and derivation, as in line_patterns intra_P but over multiple params,
    - x param div_comp: if internal compression: rm * D * L, * external compression: PP.L * L-proportional coef?
    - form_par_P if param Match | x_param Contrast: diff (D_param, ave_D_alt_params: co-derived co-vary? neg val per P, else delete?
    form_PPd: dP = dL + dM + dD  # -> directional PPd, equal-weight params, no rdn?
    if comp I -> dI ~ combined d_derivatives, then project ave_d?
    '''

    for PP in PPm_:
        if len(PP.P_) > 8 and PP.mP + PP.M > ave_PPM:
            # calls rnd_derP:
            sub_PPm_ = rng_search(PP.P_, (ave_M + PP.M / len(PP.P_)) / 2 * rdn)  # ave_M is average of local and global match

            return sub_PPm_

# below is likely obsolete:

def rng_search(P_, ave):
    comb_layers = []  # if recursive only
    sub_PPm_ = []
    rderP_ = []

    for i, P in enumerate(P_):
        neg_M = vmP = sign = _smP = neg_L = 0

        for j, _P in enumerate(P_[i + 2:]):  # i+2: skip previously compared adjacent Ps, i+3 for sparse comp?
            # variable-range comp, no last-P displacement, just shifting first _P
            if P.M * (neg_L/P.L * ave_rM) + neg_M > ave:  # search while net_M > ave

                rderP, _L, _smP = comp_P(P, _P, neg_M, neg_L)
                sign, vmP, neg_M, neg_L, P = rderP.sign, rderP.mP, rderP.neg_M, rderP.neg_L, rderP.P
                if sign:
                    P_[i + 1 + j]._smP = True  # backward match per compared _P
                    rderP_.append(rderP)
                    break  # nearest-neighbour search is terminated by first match
                else:
                    neg_M += vmP  # accumulate contiguous miss: negative mP
                    neg_L += _L   # accumulate distance to match
                    if j == len(P_):
                        # last P is a singleton derP, derivatives are ignored:
                        rderP_.append(CderP(sign=sign or _smP, mP=vmP, neg_M=neg_M, neg_L=neg_L, P=P ))
            else:
                rderP_.append(CderP(sign=sign or _smP, mP=vmP, neg_M=neg_M, neg_L=neg_L, P=P))
                # sign is ORed bilaterally, negative for singleton derPs only
                break  # neg net_M: stop search

        sub_PPm_ = form_PP_(rderP_, fPd=False)  # cluster derPs into PPms by the sign of mP, same form_PPm_?

    return sub_PPm_

'''
def form_Pp_(derP_, fPd):  # this is a draft, please check
    # rdn = layer0_rdn[param_name]
    param_names = derP_[0].layer1.key  #?
    for param_name in param_names:
        if fPd:
            derP_[0].layer1.Ppd_[0] = CP(sign = derP_[0].layer1[param_name].d > 0)  # not sure this is correct?
        else:
            derP_[0].layer1.Ppm_[0] = CP(sign = derP_[0].layer1[param_name].m > 0)
    for derP in derP_:
        for param_name in param_names:  # this is not correct, we two loops
            if fPd:
                _sign = derP_[0].layer1.param_name.Ppd_[0].sign
                sign = derP.layer1[param_name].d > 0
            else:
                _sign = derP_[0].layer1.param_name.Ppm_[0].sign
                sign = derP.layer1[param_name].m > 0
            for derP in derP_[1:]:
                d = derP.layer1[param_name].d
                m = derP.layer1[param_name].m
                if fPd: sign = d > 0
                else:   sign = m > 0
            if sign != _sign:
                if fPd: derP.layer1[param_name].Ppd_.append(Pp)
                else:   derP.layer1[param_name].Ppm_.append(Pp)
                Pp = CP(sign=_sign)
            else:
                # accumulate Pp params
                Pp.L += 1
                Pp.D += d
                Pp.M += m
                Pp.d_.append(d)
                Pp.m_.append(m)
    merge_Pp_Kelvin(derP_, fPd)
'''

def merge_Pp_Kelvin(derP_, fPd):  # merge low value Pps into negative Pp
    neg_Pp_ = []
    remove_index = []

    for i, derP in enumerate(derP_):
        for param_name in derP.layer1:
                if fPd: Pp_ = derP.layer1[param_name].Ppm_
                else:   Pp_ = derP.layer1[param_name].Ppd_

                for j, Pp in enumerate(Pp_):
                    if fPd: Pparam = Pp.M
                    else:   Pparam = Pp.D
                    if abs(Pparam) < ave:  # get low value negative Pps
                        neg_Pp_.append(Pp)
                        remove_index.append(j)
                if neg_Pp_:
                    _neg_Pp = neg_Pp_[0]
                    for i, neg_Pp in enumerate(neg_Pp_, start=1):
                        _neg_Pp.accum_from(neg_Pp)  # accumulate all neg_Pps
                    if fPd:
                        derP.layer1[param_name].Ppm_[0] = _neg_Pp  # replace first Pp in Pp_ with accumulated neg_Pp
                    else:
                        derP.layer1[param_name].Ppd_[0] = _neg_Pp
                    _neg_Pp = None

                for index in sorted(remove_index, reverse=True):  # remove all merged Ppd from original Pp_
                    if fPd:
                        del derP.layer1[param_name].Ppm_[index]
                    else:
                        del derP.layer1[param_name].Ppd_[index]
                remove_index = []
