import numpy as np
from collections import deque, defaultdict
from copy import deepcopy, copy
from itertools import zip_longest, combinations
from typing import List, Tuple
from .classes import add_, sub_, acc_, get_match
from .filters import ave, ave_dI, aves, P_aves, PP_aves
from .slice_edge import comp_angle
from utils import box2slice, accum_box, sub_box2box

'''
Vectorize is a terminal fork of intra_blob.

comp_slice traces edge axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These are low-M high-Ma blobs, vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)
-
Vectorization is clustering of parameterized Ps + their derivatives (derPs) into PPs: patterns of Ps that describe edge blob.
This process is a reduced-dimensionality (2D->1D) version of cross-comp and clustering cycle, common across this project.

PP clustering in vertical (along axis) dimension is contiguous and exclusive because 
Ps are relatively lateral (cross-axis), their internal match doesn't project vertically. 

Primary clustering by match between Ps over incremental distance (rng++), followed by forming overlapping
Secondary clusters of match of incremental-derivation (der++) difference between Ps. 

As we add higher dimensions (3D and time), this dimensionality reduction is done in salient high-aspect blobs
(likely edges in 2D or surfaces in 3D) to form more compressed "skeletal" representations of full-dimensional patterns.

comp_slice traces edge blob axis by cross-comparing vertically adjacent Ps: slices across edge blob, along P.G angle.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)

Connectivity in P_ is traced through root_s of derts adjacent to P.dert_, possibly forking. 
len prior root_ sorted by G is root.rdn, to eval for inclusion in PP or start new P by ave*rdn
'''

  # root function:
def der_recursion(root, PP, fd=0):  # node-mediated correlation clustering: keep same Ps and links, increment link derH, then P derH in sum2PP

    if fd:  # add prelinks per P if not initial call:
        # PP[2] is P_
        for P in PP[2]: P[-1] += [unpack_last_link_(P[-1])]  # P[-1] = P.link_

    rng_recursion(PP, rng=1, fd=fd)  # extend PP.link_, derHs by same-der rng+ comp
    form_PP_t(PP, PP[2], irdn=PP[7][1])  # der+ is mediated by form_PP_t
    if root: root.fback_ += [[PP.derH, PP.valt, PP.rdnt]]  # feedback from PPds


def rng_recursion(PP, rng=1, fd=0):  # similar to agg+ rng_recursion, but contiguously link mediated, because

    iP_ = PP[2]  # PP[2] = PP.P_
    while True:
        P_ = []; V = 0
        for P in iP_:
            if not P[-1]: continue  # P[-1] is P.link_
            prelink_ = []  # new prelinks per P
            _prelink_ = P[-1].pop()  # old prelinks per P (P[-1] is P.link_)
            for _link in _prelink_:
                _P = _link._P if fd else _link
                # P[3] is derH, derH[1] is H
                if len(_P[3][1])!= len(P[3][1]): continue  # compare same der layers only
                dy,dx = np.subtract(_P[10], P[10])  # P[10] is P.yx
                distance = np.hypot(dy,dx)  # distance between P midpoints, /= L for eval?
                if distance < rng:  # | rng * ((P.val+_P.val) / ave_rval)?
                    mlink = comp_P(_link if fd else [_P,P, distance,[dy,dx]], fd)  # return link if match
                    if mlink:
                        V += mlink[4][0]  # unpack last link layer: (link[4] is link.vt)
                        link_ = P[-1][-1] if PP[10] and PP[10][0] == "derH" else P[-1] # der++ if derH.depth==1
                        if rng > 1:
                            if rng == 2: link_[:] = [link_[:]]  # link_ -> link_H
                            if len(link_) < rng: link_ += [[]]  # new link_
                            link_ = link_[-1]  # last rng layer
                        link_ += [mlink]
                        prelink_ += unpack_last_link_(_P[-1][:-1])  # get last link layer, skip old prelinks
            if prelink_:
                if not fd: prelink_ = [link[2] for link in prelink_]  # prelinks are __Ps, else __links
                P[-1] += [prelink_]  # temporary prelinks
                P_ += [P]  # for next loop
        rng += 1
        if V > ave * len(P_) * 6:  #  implied val of all __P_s, 6: len mtuple
            iP_ = P_
        else:
            for P in P_: P[-1].pop()
            break
    PP[12]=rng
    '''
    der++ is tested in PPds formed by rng++, no der++ inside rng++: high diff @ rng++ termination only?
    '''

def comp_P(link, fd):

    if fd: _P, P = link[2], link[1]  # in der+ 
    else:  _P, P, S, A = link  # list in rng+
    rn = len(_P[6]) / len(P[6])  # P[6] is P.dert_

    if _P[3][1] and P[3][1]:  # P[3][1] is P.derH.H
        # der+: append link derH, init in rng++ from form_PP_t
        derLay, vt,rt,_ = comp_derH(_P.derH, P.derH, rn=rn)  # += fork rdn
        aveP = P_aves[1]
        fd=1
    else:
        # rng+: add link derH
        mtuple, dtuple = comp_ptuple(_P[1], P[1], rn)  # P[1] is P.ptuple
        vt = [sum(mtuple), sum(abs(d) for d in dtuple)]
        rt = [1+(vt[1]>vt[0]), 1+(1-(vt[1]>vt[0]))]  # or rdn = Dval/Mval?
        aveP = P_aves[0]
        fd=0
    if vt[0] > aveP*rt[0]:  # always rng+
        if fd:
            derH = link[3]
            if derH[0] == "dertv":  # add nesting dertv-> derH:
                H, valt, rdnt, dect = derH[1:]
                link[3][1] = [["derH", deepcopy(H),copy(valt),copy(rdnt),copy(dect)]]  # link[3] is link.derH
            derH[1] += derLay; link[4]=np.add(link[4],vt); link[5]=np.add(link[5],rt)
        else:
            # type, H, valt, rdnt, dect
            derH = ["dertv", [mtuple, dtuple], vt, rt, [0,0]]  # dertv
            # type, P, _P, derH, vt, rt, S, A, roott
            link = ["derP", P,_P, derH, copy(vt), copy(rt), S, A, [[],[]]]

        return link


def form_PP_t(root, P_, irdn):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    for fd in 0,1:
        P_Ps = []; Link_ = []
        for P in P_:  # not PP.link_: P uplinks are unique, only G links overlap
            Ps = []
            for derP in unpack_last_link_(P[-1]):  # P[-1] is link_
                # derP[2] is _P
                Ps += [derP[2]]; Link_ += [derP]  # not needed for PPs?
            P_Ps += [Ps]  # aligned with P_
        inP_ = []  # clustered Ps and their val,rdn s for all Ps
        for P in root[2]:  # root[2] is P_
            if P in inP_: continue  # already packed in some PP
            cP_ = [P]  # clustered Ps and their val,rdn s
            if P in P_:
                perimeter = deque(P_Ps[P_.index(P)])  # recycle with breadth-first search, up and down:
                while perimeter:
                    _P = perimeter.popleft()
                    if _P in cP_: continue
                    cP_ += [_P]
                    if _P in P_:
                        perimeter += P_Ps[P_.index(_P)] # append linked __Ps to extended perimeter of P
            PP = sum2PP(root, cP_, Link_, irdn, fd)
            PP_t[fd] += [PP]  # no if Val > PP_aves[fd] * Rdn:
            inP_ += cP_  # update clustered Ps

    for PP in PP_t[1]:  # eval der+ / PPd only, after form_PP_t -> P.root
        if PP.Vt[1] * len(PP.link_) > PP_aves[1] * PP.Rt[1]:
            # node-mediated correlation clustering:
            der_recursion(root, PP, fd=1)
        if root.fback_:
            feedback(root)  # after der+ in all nodes, no single node feedback

    root.node_ = PP_t  # nested in der+, add_alt_PPs_?


def sum2PP(root, P_, derP_, irdn, fd):  # sum links in Ps and Ps in PP

    # type, root, P_, link_, fd, ptuple, Vt, Rt, Dt, box, derH, fback_, rng, ext, mask__
    ptuple= [0,0,0,0,[0,0],0]
    derH = ["derH", [], [0,0], [1,1], [0,0]]
    PP = ["PP", root, P_, [], fd, ptuple, [0,0], [1,1], [0,0], [0,0,0,0], derH, [], root[12] + 1, [], None ]

    # += uplinks:
    S,A = 0, [0,0]
    for derP in derP_:
        if derP[1] not in P_ or derP[2] not in P_: continue
        # derP[3][5] is derP.derH.rdnt
        derP[3][5][fd] += irdn
        # derP[1][3] is derP.P.derH, derP[3] is derH
        derP[1][3] = add_(derP[1][3], derP[3])
        # derP[2][3][5] is derP._P.derH.rdnt
        derP[2][3][5][fd] += irdn
        # derP[2][3] is derP._P.derH
        derP[2][3] = add_(derP[2][3], negate(derP[3]))  # reverse d signs downlink
        PP[3] += [derP]; derP[8][fd] = PP  # derP[8] is roott
        PP[6] = np.add(PP[6],derP[4])
        PP[7] = np.add(np.add(PP[7],derP[5]), [irdn,irdn])
        derP[7] = np.add(A,derP[7]); S += derP[6]  # derP[6] is S, # derP[7] is A
    PP[13] = [len(P_), S, A]  # all from links

    # += Ps:
    celly_,cellx_ = [],[]
    for P in P_:
        PP[5] = add_(PP[5], P[1])  # PP[5] is ptuple, P[1] is ptuple
        PP[10] = add_(PP[10], P[3])  # PP[10] is derH, P[3] is derH
        for y,x in P[7]:  # P[7] is P.cells
            PP[9] = accum_box(PP[9], y, x); celly_+=[y]; cellx_+=[x]
    # pixmap:
    y0,x0,yn,xn = PP[9]
    PP[14] = np.zeros((yn-y0, xn-x0), bool)  # PP[14] is mask__
    celly_ = np.array(celly_); cellx_ = np.array(cellx_)
    PP[14][(celly_-y0, cellx_-x0)] = True

    return PP


def negate(instance):
    pass

def feedback(root):  # in form_PP_, append new der layers to root PP, single vs. root_ per fork in agg+

    derH, valt, rdnt = CderH(),[0,0],[0,0]
    while root.fback_:
        _derH, _valt, _rdnt = root.fback_.pop(0)
        derH += _derH; acc_(valt,_valt); acc_(rdnt,_rdnt)

    root.derH += derH; add_(root.valt,_valt); add_(root.rdnt,_rdnt)

    if isinstance(root.root, z):  # skip if root is Edge
        rroot = root.root  # single PP.root, can't be P
        fback_ = rroot.fback_
        # rroot.node_ may get empty list when root is PP because node_ is packed in P_
        # it wil be updated to node_t at the endof form_PP_t only, which is after feedback
        node_ = rroot.node_[1] if isinstance(rroot.node_[0],list) else rroot.node_  # node_ is updated to node_t in sub+
        fback_ += [(derH, valt, rdnt)]
        if fback_ and (len(fback_)==len(node_)):  # all nodes terminated and fed back
            feedback(rroot)  # sum2PP adds derH per rng, feedback adds deeper sub+ layers


def comp_dtuple(_ptuple, ptuple, rn, fagg=0):

    mtuple, dtuple = [],[]
    if fagg: Mtuple, Dtuple = [],[]

    for _par, par, ave in zip(_ptuple, ptuple, aves):  # compare ds only
        npar = par * rn
        mtuple += [get_match(_par, npar) - ave]
        dtuple += [_par - npar]
        if fagg:
            Mtuple += [max(abs(par),abs(npar))]
            Dtuple += [abs(_par)+abs(npar)]
    ret = [mtuple, dtuple]
    if fagg: ret += [Mtuple, Dtuple]
    return ret

def comp_ptuple(_ptuple, ptuple, rn, fagg=0):  # 0der params

    _I, _G, _M, _Ma, (_Dy, _Dx), _L = _ptuple
    I, G, M, Ma, (Dy, Dx), L = ptuple

    dI = _I - I*rn;  mI = ave_dI - dI
    dG = _G - G*rn;  mG = min(_G, G*rn) - aves[1]
    dL = _L - L*rn;  mL = min(_L, L*rn) - aves[2]
    dM = _M - M*rn;  mM = get_match(_M, M*rn) - aves[3]  # M, Ma may be negative
    dMa= _Ma- Ma*rn; mMa = get_match(_Ma, Ma*rn) - aves[4]
    mAngle, dAngle = comp_angle((_Dy,_Dx), (Dy,Dx))

    mtuple = [mI, mG, mM, mMa, mAngle-aves[5], mL]
    dtuple = [dI, dG, dM, dMa, dAngle, dL]
    ret = [mtuple, dtuple]
    if fagg:
        Mtuple = [max(_I,I), max(_G,G), max(_M,M), max(_Ma,Ma), 2, max(_L,L)]
        Dtuple = [abs(_I)+abs(I), abs(_G)+abs(G), abs(_M)+abs(M), abs(_Ma)+abs(Ma), 2, abs(_L)+abs(L)]
        ret += [Mtuple, Dtuple]
    return ret

def comp_ptuple_generic(_ptuple, ptuple, rn):  # 0der

    mtuple, dtuple, Mtuple = [],[],[]
    # _n, n = _ptuple, ptuple: add to rn?
    for i, (_par, par, ave) in enumerate(zip(_ptuple, ptuple, aves)):
        if isinstance(_par, list) or isinstance(_par, tuple):
             m,d = comp_angle(_par, par)
             maxv = 2
        else:  # I | M | G L
            npar= par*rn  # accum-normalized param
            d = _par - npar
            if i: m = min(_par,npar)-ave
            else: m = ave-abs(d)  # inverse match for I, no mag/value correlation
            maxv = max(_par, par)
        mtuple+=[m]
        dtuple+=[d]
        Mtuple+=[maxv]
    return [mtuple, dtuple, Mtuple]

# not revised
def comp_derH(_derH, derH, rn=1, fagg=0):  # derH is a list of der layers or sub-layers, each = ptuple_tv

    Ht = []
    fptuple = 0
    if _derH.typ == "ptuple":
        Ht = [[_derH], [derH]]; fptuple = 1
    else:
        for derH in [_derH, derH]:  # init H is dertuplet, convert to dertv_ (permanent conversion in sum2PP):
            Ht += [derH.H] if (not isinstance(derH.H[0], list)) else [[CderH(H=derH.H, valt=copy(derH.valt), rdnt=copy(derH.rdnt), dect=copy(derH.dect), depth=0)]]
    derLay = []; Vt,Rt,Dt = [0,0],[0,0],[0,0]

    for _lay, lay in zip(Ht[0], Ht[1]):
        if fptuple:  # comp_ptuple
            der = comp_ptuple(_lay, lay, rn=rn, fagg=fagg)
        else:  # comp_derH
            der = comp_dtuple(_lay.H[1], lay.H[1], rn=rn, fagg=fagg)

        if fagg: mtuple, dtuple, Mtuple,Dtuple = der
        else:    mtuple, dtuple = der

        valt = [sum(mtuple),sum(abs(d) for d in dtuple)]
        rdnt = [valt[1] > valt[0], valt[1] <= valt[0]]
        dect = [0,0]
        if fagg:
            for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
                for (par, max, ave) in zip(ptuple, Ptuple, aves):  # different ave for comp_dtuple
                    if fagg:
                        if fd: dect[1] += abs(par)/ abs(max) if max else 1
                        else:  dect[0] += (par+ave)/ (max+ave) if max else 1
            if fagg:
                dect[0] = dect[0]/6; dect[1] = dect[1]/6  # ave of 6 params

        Vt = np.add(Vt,valt); Rt = np.add(Rt,rdnt)
        if fagg: Dt = np.divide(np.add(Dt,dect),2)
        derLay += [CderH(H=[mtuple,dtuple], valt=valt,rdnt=rdnt,dect=dect, depth=0)]  # dertvs

    return derLay[0] if fptuple else derLay, Vt,Rt,Dt  # to sum in each G Et


# replace by += overload:

def sum_derH(T, t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH, Valt, Rdnt = T; derH, valt, rdnt = t
    for i in 0,1:
        Valt[i] += valt[i]
        Rdnt[i] += rdnt[i] + base_rdn
    DerH[:] = [
        # sum der layers, dertuple is mtuple | dtuple, fneg*i: for dtuple only:
        [ sum_dertuple(Mtuple, mtuple, fneg=0), sum_dertuple(Dtuple, dtuple, fneg=fneg) ]
        for [Mtuple, Dtuple], [mtuple, dtuple]
        in zip_longest(DerH, derH, fillvalue=[[0,0,0,0,0,0],[0,0,0,0,0,0]])  # mtuple,dtuple
    ]

def sum_dertuple(Ptuple, ptuple, fneg=0):
    _I, _G, _M, _Ma, _A, _L = Ptuple
    I, G, M, Ma, A, L = ptuple
    if fneg: Ptuple[:] = [_I-I, _G-G, _M-M, _Ma-Ma, _A-A, _L-L]
    else:    Ptuple[:] = [_I+I, _G+G, _M+M, _Ma+Ma, _A+A, _L+L]
    return   Ptuple

def sum_derH_generic(T, t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH, Valt, Rdnt = T; derH, valt, rdnt = t
    Rdnt += rdnt + base_rdn
    Valt += valt
    if DerH:
        for Layer, layer in zip_longest(DerH,derH, fillvalue=[]):
            if layer:
                if Layer:
                    for i in 0,1:
                        sum_dertuple(Layer[0][i], layer[0][i], fneg and i)  # ptuplet, only dtuple is directional: needs fneg
                        Layer[1][i] += layer[1][i]  # valt
                        Layer[2][i] += layer[2][i] + base_rdn  # rdnt
                else:
                    DerH += [deepcopy(layer)]
    else:
        DerH[:] = deepcopy(derH)


def unpack_last_link_(link_):  # unpack last link layer

    while link_ and isinstance(link_[-1], list): link_ = link_[-1]
    return link_
