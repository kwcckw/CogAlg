import numpy as np
from collections import deque, defaultdict
from copy import deepcopy, copy
from itertools import zip_longest, combinations
from math import inf
from typing import List, Tuple
from .classes import add_, comp_, negate, get_match, CPP, Cptuple, CderP, z
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
(likely edges in 2D or surfaces in 3D) to form more compressed 'skeletal' representations of full-dimensional patterns.

comp_slice traces edge blob axis by cross-comparing vertically adjacent Ps: slices across edge blob, along P.G angle.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)

Connectivity in P_ is traced through root_s of derts adjacent to P.dert_, possibly forking. 
len prior root_ sorted by G is root.rdn, to eval for inclusion in PP or start new P by ave*rdn
'''

  # root function:
def der_recursion(root, PP, fd=0):  # node-mediated correlation clustering: keep same Ps and links, increment link derH, then P derH in sum2PP

    (depth, Et, n, pp_id), et, (ptuple, He), (P_, node_) = PP[:4]
    if fd:  # add prelinks per P if not initial call:
        for P in P_: P[3] += [copy(unpack_last_link_(P[3]))]

    rng_recursion(PP, rng=1, fd=fd)  # extend PP.link_, derHs by same-der rng+ comp

    form_PP_t(PP, P_, iRt = Et[2:4] if Et else [0,0])  # der+ is mediated by form_PP_t
    if root: root.fback_ += [[He, et]]  # feedback from PPds


def rng_recursion(PP, rng=1, fd=0):  # similar to agg+ rng_recursion, but contiguously link mediated, because

    iP_ = PP[3][0]  # PP[3] is (P_, node_)
    while True:
        P_ = []; V = 0
        for P in iP_:
            (depth, Et, n, p_id), (ptuple, He), dert_,link_, yx, axis, cells =  P
            if not link_: continue
            prelink_ = []  # new prelinks per P
            _prelink_ = link_.pop()  # old prelinks per P
            for _link in _prelink_:
                _P = _link[4] if fd else _link
                (_depth, _Et, _n, _p_id), (_ptuple, _He), _dert_,_link_, _yx, _axis, _cells =  P = _P
                dy,dx = np.subtract(_yx, yx)
                distance = np.hypot(dy,dx)  # distance between P midpoints, /= L for eval?
                if distance < rng:  # | rng * ((P.val+_P.val) / ave_rval)?
                    mlink = comp_P(_link if fd else [_P,P, distance,[dy,dx]], fd)  # return link if match
                    if mlink:
                        V += mlink[0][1][0]  # unpack last link layer: ([0][1] = core's Et)
                        link_ = link_[-1] if link_ and isinstance(link_[-1], list) else link_  # der++ if PP.He[0] depth==1
                        if rng > 1:
                            if rng == 2: link_[:] = [link_[:]]  # link_ -> link_H
                            if len(link_) < rng: link_ += [[]]  # new link_
                            link_ = link_[-1]  # last rng layer
                        link_ += [mlink]
                        prelink_ += unpack_last_link_(_link_[:-1])  # get last link layer, skip old prelinks
            if prelink_:
                if not fd: prelink_ = [link[4] for link in prelink_]  # prelinks are __Ps, else __links
                link_ += [prelink_]  # temporary prelinks
                P_ += [P]  # for next loop
        rng += 1
        if V > ave * len(P_) * 6:  #  implied val of all __P_s, 6: len mtuple
            iP_ = P_
        else:
            for P in P_: P[3].pop()  # pop prelinks fromn link_
            break
    PP[-1]=rng  # both PP and edge['s last element is rng
    '''
    der++ is tested in PPds formed by rng++, no der++ inside rng++: high diff @ rng++ termination only?
    '''

def comp_P(link, fd):

    if fd: _P, P = link[2]  # in der+
    else:  _P, P, S, A = link  # list in rng+
    (depth,   Et,  n,  p_id), (ptuple,   He),  dert_, link_,  yx,  axis,  cells =  P
    (_depth, _Et, _n, _p_id), (_ptuple, _He), _dert_,_link_, _yx, _axis, _cells = _P
    rn = len(_dert_) / len(dert_)

    if _He and He:
        # der+: append link derH, init in rng++ from form_PP_t
        depth,(vm,vd,rm,rd),H, n = comp_(_He, He, rn=rn)
        rm += vd > vm; rd += vm >= vd
        aveP = P_aves[1]
    else:
        # rng+: add link derH
        H = comp_ptuple(_ptuple, ptuple, rn)
        vm = sum(H[::2]); vd = sum(abs(d) for d in H[1::2])
        rm = 1 + vd > vm; rd = 1 + vm >= vd
        aveP = P_aves[0]
        n = 1  # 6 compared params is a unit of n

    if vm > aveP*rm:  # always rng+
        if fd:
            He = link[2]
            if not He[0]: He = link[2] = [1,[*He[1]],[He]]  # nest md_ as derH
            He[1] = np.add(He[1],[vm,vd,rm,rd])
            He[2] += [[0, [vm,vd,rm,rd], H]]  # nesting, Et, H
            link[1] = [V+v for V, v in zip(link[0][1],[vm,vd, rm,rd])]
        else:
            # core = depth, Et, n, id
            core = [0, [vm,vd,rm,rd], n, 0]; He =[0,[vm,vd,rm,rd],H]; roott = [[],[]]
            link = [core,He,(_P,P),[S,A],roott]  # core, He, (_P,P), (S,A), roott

        return link


def form_PP_t(root, P_, iRt):  # form PPs of derP.valt[fd] + connected Ps val

    root_P_, root_fback_ = root[3][0], root[-2]
    PP_t = [[],[]]
    for fd in 0,1:
        P_Ps = []; Link_ = []
        for P in P_:  # not PP.link_: P uplinks are unique, only G links overlap
            (depth,Et,n,p_id), (ptuple,He), dert_, link_, yx, axis, cells = P
            Ps = []
            for derP in unpack_last_link_(link_):
                _P = derP[2][0]; Ps += [_P]; Link_ += [derP]  # not needed for PPs?
            P_Ps += [Ps]  # aligned with P_
        inP_ = []  # clustered Ps and their val,rdn s for all Ps
        for P in root_P_:
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
            PP = sum2PP(root, cP_, Link_, iRt, fd)
            PP_t[fd] += [PP]  # no if Val > PP_aves[fd] * Rdn:
            inP_ += cP_  # update clustered Ps

    # section below not updated
    for PP in PP_t[1]:  # eval der+ / PPd only, after form_PP_t -> P.root
        if PP.Et[1] * len(PP.link_) > PP_aves[1] * PP.Et[3]:
            # node-mediated correlation clustering:
            der_recursion(root, PP, fd=1)
        if root_fback_:
            feedback(root)  # after der+ in all nodes, no single node feedback

    root[3][1] = PP_t  # nested in der+, add_alt_PPs_?


def sum2PP(root, P_, derP_, iRt, fd):  # sum links in Ps and Ps in PP

    # core, et, (ptuple, He), (P_,node_),link_, (ext,box,mask__,area), root, fd, fback_, ,rng
    pp_depth=1; pp_Et = [0,0,1,1]; pp_n =0; pp_id=0;  # core
    pp_et=[0,0,1,1];pp_ptuple = [0,0,0,0, [0,0],0];pp_He = [];pp_P_ = P_;pp_node_=[];pp_link_=[];pp_ext=[];pp_area=0;pp_root=[];pp_fd=fd;pp_fback_=[];pp_rng = root[-1] + 1

    # += uplinks:
    S,A = 0, [0,0]
    for derP in derP_:
        (derP_depth, derP_et, derP_n, derP_id), derP_He, (_P,P), (derP_S,derP_A), derP_roott = derP
        if P not in P_ or _P not in P_: continue
        if derP_He:
            P_He, _P_He = P[1][1], _P[1][1]
            add_(P_He, derP_He, iRt)
            add_(_P_He, negate(deepcopy(derP_He)), iRt)
        pp_link_ += [derP]
        pp_Et = [V+v for V,v in zip(pp_Et, derP_et)]
        pp_Et[2:4] = [R+ir for R,ir in zip(pp_Et[2:4], iRt)]
        pp_n += derP_n
        A = np.add(A,derP_A); S += derP_S
    if S: pp_ext = [len(P_), S , A]  # all from links (skip 0 S when PP is single P's PP)

    # += Ps:
    for P in P_:
        (depth,Et,n,p_id), (ptuple,He), dert_, link_, yx, axis, cells = P
        pp_area += len(dert_) 
        pp_ptuple = [ppar + par if not isinstance(ppar, list) else [ppar[0]+par[0], ppar[1]+par[1]]  for ppar, par in zip(pp_ptuple,ptuple)]
        if He:
            add_(pp_He, He)
            pp_et = [V+v for V, v in zip(pp_et, He[1])]  # we need to sum et from P too? Else they are always empty

    # below should be not needed, but we probably need yx_ now?
    '''
    # pixmap:
    y0,x0,yn,xn = pp_box
    pp_mask__ = np.zeros((yn-y0, xn-x0), bool)
    celly_ = np.array(celly_); cellx_ = np.array(cellx_)
    pp_mask__[(celly_-y0, cellx_-x0)] = True
    '''
    
    # core, et, (ptuple, He), (P_,node_),link_, (ext,box,mask__,area), root, fd, fback_ ,rng
    PP = [[pp_depth,pp_Et,pp_n,pp_id], pp_et, [pp_ptuple, pp_He], [pp_P_,pp_node_],pp_link_,[pp_ext,pp_area], pp_root, pp_fd, pp_fback_ ,pp_rng]
    for derP in derP_: derP[-1][fd] = PP  # update root  

    return PP


def feedback(root):  # in form_PP_, append new der layers to root PP, single vs. root_ per fork in agg+

    HE, eT= deepcopy(root.fback_.pop(0))
    while root.fback_:
        He, et = root.fback_.pop(0)
        eT = [V+v for V,v in zip(eT, et)]
        add_(HE, He)
    add_(root.He, HE if HE[0] else HE[2][-1])  # sum md_ or last md_ in H
    root.et = [V+v for V,v in zip_longest(root.et, eT, fillvalue=0)]  # fillvalue to init from empty list

    if root.typ != 'edge':  # skip if root is Edge
        rroot = root.root  # single PP.root, can't be P
        fback_ = rroot.fback_
        node_ = rroot.node_[1] if rroot.node_ and isinstance(rroot.node_[0],list) else rroot.P_  # node_ is updated to node_t in sub+
        fback_ += [(HE, eT)]
        if fback_ and (len(fback_)==len(node_)):  # all nodes terminated and fed back
            feedback(rroot)  # sum2PP adds derH per rng, feedback adds deeper sub+ layers


def comp_ptuple(_ptuple, ptuple, rn, fagg=0):  # 0der params

    _I, _G, _M, _Ma, (_Dy, _Dx), _L = _ptuple
    I, G, M, Ma, (Dy, Dx), L = ptuple

    dI = _I - I*rn;  mI = ave_dI - dI
    dG = _G - G*rn;  mG = min(_G, G*rn) - aves[1]
    dL = _L - L*rn;  mL = min(_L, L*rn) - aves[2]
    dM = _M - M*rn;  mM = get_match(_M, M*rn) - aves[3]  # M, Ma may be negative
    dMa= _Ma- Ma*rn; mMa = get_match(_Ma, Ma*rn) - aves[4]
    mAngle, dAngle = comp_angle((_Dy,_Dx), (Dy,Dx))

    ret = [mI,dI,mG,dG,mM,dM,mMa,dMa,mAngle-aves[5],dAngle,mL,dL]

    if fagg:  # add norm m,d: ret = [ret, Ret]
        # max possible m,d per compared param
        Ret = [max(_I,I), abs(_I)+abs(I), max(_G,G),abs(_G)+abs(G), max(_M,M),abs(_M)+abs(M), max(_Ma,Ma),abs(_Ma)+abs(Ma), 1,.5, max(_L,L),abs(_L)+abs(L)]
        mval, dval = sum(ret[::2]),sum(ret[1::2])
        mrdn, drdn = dval>mval, mval>dval
        mdec, ddec = 0, 0
        for fd, (ptuple,Ptuple) in enumerate(zip((ret[::2],ret[1::2]),(Ret[::2],Ret[1::2]))):
            for i, (par, maxv, ave) in enumerate(zip(ptuple, Ptuple, aves)):
                # compute link decay coef: par/ max(self/same)
                if fd: ddec += abs(par)/ abs(maxv) if maxv else 1
                else:  mdec += (par+ave)/ (maxv+ave) if maxv else 1
        mdec /= 6; ddec /= 6  # ave of 6 params
        ret = [mval, dval, mrdn, drdn, mdec, ddec], ret
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

def unpack_last_link_(link_):  # unpack last link layer

    while link_ and link_[-1] and link_[-1][0][0] != 0:  link_ = link_[-1]  # check 0, which is the depth of link
    return link_