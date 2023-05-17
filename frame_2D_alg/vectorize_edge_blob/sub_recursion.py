from copy import copy, deepcopy
import numpy as np
from .filters import PP_aves, ave, ave_nsub, P_aves, G_aves
from .classes import CP, CQ, CderP, CPP
from .comp_slice import comp_P, form_PP_t, sum_derH
from .agg_convert import agg_recursion_eval

def sub_recursion_eval(root, PP_, fd):  # for PP or blob

    for PP in PP_:  # fd = _P.valt[1]+P.valt[1] > _P.valt[0]+_P.valt[0]  # if exclusive comp fork per latuple|vertuple?
        # fork val, rdn:
        val = PP.valt[fd]
        ave = PP_aves[fd] * PP.rdnt[fd]
        if val > ave and len(PP.P__) > ave_nsub:
            sub_recursion(PP, fd)  # comp_der | comp_rng in PPs -> param_layer, sub_PPs
        if isinstance(root, CPP):
            for fd in 0,1:
                root.valt[fd] += PP.valt[fd]; root.rdnt[fd] += PP.rdnt[fd]  # add rdn?
        else:  # root is Blob
            if fd: root.G += PP.valt[1-fd]
            else:  root.M += PP.valt[fd]


def sub_recursion(PP, fd):  # evaluate PP for rng+ and der+, add layers to select sub_PPs

    P__ = comp_der(PP.P__) if fd else comp_rng(PP.P__, PP.rng+1)   # returns top-down
    PP.rdnt[fd] += PP.valt[fd] > PP.valt[1-fd]
    # link Rdn += PP rdn?
    cP__ = [copy(P_) for P_ in P__]
    sub_PPm_, sub_PPd_ = form_PP_t(PP, cP__, base_rdn=PP.rdnt[fd])
    PP.P__ = [sub_PPm_, sub_PPd_]  # node_t

    for fd, sub_PP_ in enumerate([sub_PPm_, sub_PPd_]):
        if PP.valt[fd] > ave * PP.rdnt[fd]:
            sub_recursion_eval(PP, sub_PP_, fd=fd)
        else:
            for sub_PP in sub_PP_: sub_PP.fterm = 1  # we need to terminate sub_PP here? Else their fterm will be 0 because they won't run through sub_recursion_eval
            feedback(PP, fd)  # update derH, valt, rdnt, add rngH: rng lays are not represented in derH?
        if PP.valt[fd] > ave * PP.rdnt[fd]:  # adjusted by sub+, ave*agg_coef?
            agg_recursion_eval(PP, copy(sub_PP_), fd=fd)  # comp sub_PPs, form intermediate PPs
        else:
            feedback(PP, fd)  # add aggH: levs of rngH( derH?

# mderP * (ave_olp_L / olp_L)? or olp(_derP._P.L, derP.P.L)? gap: neg_olp, ave = olp-neg_olp?
# __Ps in rng+ are mediated by PP.rng layers of _Ps:

def comp_rng(iP__, rng):  # form new Ps and links in rng+ PP.P__, switch to rng+n to skip clustering?

    P__ = []
    for iP_ in reversed(iP__[:-rng]):  # lower compared row, follow uplinks, no uplinks in last rng rows
        P_ = []
        for P in iP_:
            link_, link_m, link_d = [],[],[]  # for new P
            Valt = [0,0]; Rdnt = [0,0]
            DerH = [[[],[]]]  # Mt,Dt
            for iderP in P.link_t[0]:  # mlinks
                _P = iderP._P
                for _derP in _P.link_t[0]:  # next layer of mlinks
                    __P = _derP._P  # next layer of Ps
                    comp_P(P,__P, link_,link_m,link_d, Valt, Rdnt, DerH, fd=0)
            if Valt[0] > P_aves[0] * Rdnt[0]:
                # add new P in rng+ PP:
                P_ += [CP(ptuple=deepcopy(P.ptuple), derH=[DerH], dert_=copy(P.dert_), fds=copy(P.fds)+[0], x0=P.x0, y0=P.y0,
                      valt=Valt, rdnt=Rdnt, link_=link_, link_t=[link_m,link_d])]
        P__+= [P_]
    return P__

def comp_der(iP__):  # form new Ps and links in rng+ PP.P__, extend their link.derH, P.derH, _P.derH

    P__ = []
    for iP_ in reversed(iP__[:-1]):  # lower compared row, follow uplinks, no uplinks in last row
        P_ = []
        for P in iP_:
            DerH, DerLay = [],[]  # new lower and last layer
            link_, link_m, link_d = [],[],[]  # for new P
            Valt = [0,0]; Rdnt = [0,0]
            for iderP in P.link_t[1]:  # dlinks
                if iderP._P.link_t[1]:  # else no _P links and derH to compare
                    _P = iderP._P
                    comp_P(_P, P, link_,link_m,link_d, Valt, Rdnt, DerH, fd=1, derP=iderP, DerLay=DerLay)
            if Valt[1] > P_aves[1] * Rdnt[1]:
                # add new P in der+ PP:
                P_ += [CP(ptuple=deepcopy(P.ptuple), derH=DerH+[DerLay], dert_=copy(P.dert_), fds=copy(P.fds)+[1],
                          x0=P.x0, y0=P.y0, valt=Valt, rdnt=Rdnt, rdnlink_=link_, link_t=[link_m,link_d])]
        P__+= [P_]
    return P__


def sum2PPP(qPPP, base_rdn, fd):  # sum PP_segs into PP
    pass

# draft, valid for sub+ only now 
def feedback(root, fd):  # bottom-up update root.derH, breadth-first, separate for each fork?

    ave = G_aves[root.fds[-1]]
    fbV = ave+1
    root.fterm = 1  # root need to be terminated here? Because if fbV > ave is false, they will never get terminated

    while root and fbV > ave:
        if isinstance(root, CPP):  # root is PP
            PP_ = root.P__[fd]
        else:  # root is blob
            PP_ = [root.PPm_,root.PPd_][fd]
        if all([node.fterm for node in PP_ ]):  # forward was terminated in all nodes
            fbval, fbrdn = 0,1
            for node in PP_:
                if isinstance(node.P__[0], CPP):  # the last terminated sub_PP doesn't run through sub+, their P__ is still Ps, instead of PPs
                    for sub_node in node.P__[fd]:
                        sum_derH(root.derH, sub_node.derH)  # node.derH is summed in root.derH in sum2PP?
                        fbval += sub_node.valt[fd]
                        fbrdn += sub_node.rdnt[fd]
            fbV = fbval/fbrdn
            root = root.root
        else:
            break