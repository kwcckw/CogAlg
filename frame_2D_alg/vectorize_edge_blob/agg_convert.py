'''
Agg_recursion eval and PP->graph conversion
'''

import numpy as np
from .agg_recursion import Cgraph, agg_recursion
from copy import copy, deepcopy
from .classes import CP, CderP, CPP
from .filters import PP_vars, PP_aves, ave_nsubt, ave_agg, med_decay
from .comp_slice import sum_derH

# move here temporary, for debug purpose
# not fully updated
def agg_recursion_eval(blob, PP_, fd):

    for i, PP in enumerate(PP_):
        converted_graph  = PP2graph(PP, 0, fd)  # convert PP to graph
        PP_[i] = converted_graph
    converted_blob = blob2graph(blob, 0, fd)  # convert root to graph

    # use internal params?
    # with scheme of tt, we should use index 0 or 1 here with every new agg+ valtt[0] or valtt[1]?
    fork_rdnt = [1+(converted_blob.valtt[0][fd] > converted_blob.valtt[0][1-fd]), 1+(converted_blob.valtt[0][1-fd] > converted_blob.valtt[0][fd])]
    if (converted_blob.valtt[0][fd] > PP_aves[fd] * ave_agg * (converted_blob.rdntt[0][fd]+1) * fork_rdnt[fd]) \
        and len(PP_) > ave_nsubt[fd]: # and converted_blob[0].alt_rdn < ave_overlap:
        converted_blob.rdntt[0][fd] += 1
        agg_recursion(converted_blob)

# old
def frame2graph(frame, fseg, Cgraph):  # for frame_recursive

    mblob_ = frame.PPm_; dblob_ = frame.PPd_  # PPs are blobs here
    x0, xn, y0, yn = frame.box
    gframe = Cgraph(alt_plevels=CpH, rng=mblob_[0].rng, rdn=frame.rdn, x0=(x0+xn)/2, xn=(xn-x0)/2, y0=(y0+yn)/2, yn=(yn-y0)/2)
    for fd, blob_, plevels in enumerate(zip([mblob_,dblob_], [gframe.plevels, gframe.alt_plevels])):
        graph_ = []
        for blob in blob_:
            graph = PP2graph(blob, fseg, Cgraph, fd)
            sum_pH(plevels, graph.plevels)
            graph_ += [graph]
        [gframe.node_.Q, gframe.alt_graph_][fd][:] = graph_  # mblob_|dblob_, [:] to enable to left hand assignment, not valid for object

    return gframe

# tentative, will be finalized when structure in agg+ is finalized
def blob2graph(blob, fseg, fd):

    PP_ = [blob.PPm_, blob.PPd_][fd]
    x0, xn, y0, yn = blob.box
    Graph = Cgraph(fd=PP_[0].fd, rng=PP_[0].rng, id_Ht = [[0],[]], box=[(y0+yn)/2,(x0+xn)/2, y0,yn, x0,xn])
    [blob.mgraph, blob.dgraph][fd] = Graph  # update graph reference

    for i, PP in enumerate(PP_):
        graph = PP2graph(PP, fseg, fd)
        sum_derH([Graph.derHt[0], Graph.valtt[0], Graph.rdntt[0]], [graph.derHt[0], graph.valtt[0], graph.rdntt[0]], 0)  # skip index 0, external params are empty now
        graph.root = Graph
        Graph.node_ += [graph]
    Graph.id_Ht[0] += [len(Graph.derHt[0])]  # add index of derH

    return Graph

# tentative, will be finalized when structure in agg+ is finalized
def PP2graph(PP, fseg, ifd=1):

    box = [(PP.box[0]+PP.box[1]) /2, (PP.box[2]+PP.box[3]) /2] + list(PP.box)
    graph = Cgraph(derHt = [deepcopy(PP.derH), []], valtt=[copy(PP.valt), [0,0]], rdntt=[copy(PP.rdnt)], id_Ht=[[0, len(PP.derH)], []], box=box)
    return graph  # the converted graph doesn't have links yet, so init their valt with PP.valt?

# all the code below should be not needed now
# drafts:
def inpack_derH(pPP, ptuples, idx_=[]):  # pack ptuple vars in derH of pPP vars, converting macro derH -> micro derH
    # idx_: indices per lev order in derH, lenlev: 1, 1, 2, 4, 8...

    repack(pPP, ptuples[0], idx_+[0])  # single-element 1st lev
    if len(ptuples)>1:
        repack(pPP, ptuples[1], idx_+[1])  # single-element 2nd lev
        i=2; last=4
        idx = 2  # init incremental elevation = i
        while last<=len(ptuples):
            lev = ptuples[i:last]  # lev is nested, max len_sublev = lenlev-1, etc.
            inpack_derH(pPP, lev, idx_+[idx])  # add idx per sublev
            i=last; last+=i  # last=i*2
            idx+=1  # elevation in derH

def repack(pPP, ptuple, idx_):  # pack derH in elements of iderH

    for i, param_name in enumerate(PP_vars):
        par = getattr(ptuple, param_name)
        Par = pPP[i]
        if len(Par) > len(idx_):  # Par is derH of pars
            Par[-1] += [par]  # pack par in top lev of Par, added per inpack_derH recursion
        else:
            Par += [[par]]  # add new Par lev, implicitly nested in ptuples?

# temporary, not used here:
# _,_,med_valH = med_eval(link._P.link_t[fd], old_link_=[], med_valH=[], fd=fd)  # recursive += mediated link layer

def med_eval(last_link_, old_link_, med_valH, fd):  # recursive eval of mediated link layers, in form_graph only?

    curr_link_ = []; med_val = 0
    # compute med_valH, layer= val of links mediated by incremental number of nodes:

    for llink in last_link_:
        for _link in llink._P.link_t[fd]:
            if _link not in old_link_:  # not-circular link
                old_link_ += [_link]  # evaluated mediated links
                curr_link_ += [_link]  # current link layer,-> last_link_ in recursion
                med_val += np.sum(_link.valT[fd])
    med_val *= med_decay ** (len(med_valH) + 1)
    med_valH += [med_val]
    if med_val > aveB:
        # last med layer val-> likely next med layer val
        curr_link_, old_link_, med_valH = med_eval(curr_link_, old_link_, med_valH, fd)  # eval next med layer

    return curr_link_, old_link_, med_valH

# currently not used:

def sum_unpack(Q,q):  # recursive unpack of two pairs of nested sequences, to sum final ptuples

    Que,Val_,Rdn_ = Q; que,val_,rdn_ = q  # alternating rngH( derH( rngH... nesting, down to ptuple|val|rdn
    for i, (Ele,Val,Rdn, ele,val,rdn) in enumerate(zip_longest(Que,Val_,Rdn_, que,val_,rdn_, fillvalue=[])):
        if ele:
            if Ele:
                if isinstance(val,list):  # element is layer or fork
                    sum_unpack([Ele,Val,Rdn], [ele,val,rdn])
                else:  # ptuple
                    Val_[i] += val; Rdn_[i] += rdn
                    sum_ptuple(Ele, ele)
            else:
                Que += [deepcopy(ele)]; Val_+= [deepcopy(val)]; Rdn_+= [deepcopy(rdn)]

def comp_unpack(Que,que, rn):  # recursive unpack nested sequence to compare final ptuples

    DerT,ValT,RdnT = [[],[]],[[],[]],[[],[]]  # alternating rngH( derH( rngH.. nesting,-> ptuple|val|rdn

    for Ele,ele in zip_longest(Que,que, fillvalue=[]):
        if Ele and ele:
            if isinstance(Ele[0],list):
                derT,valT,rdnT = comp_unpack(Ele, ele, rn)
            else:
                # elements are ptuples
                mtuple, dtuple = comp_dtuple(Ele, ele, rn)  # accum rn across higher composition orders
                mval=sum(mtuple); dval=sum(dtuple)
                derT = [mtuple, dtuple]
                valT = [mval, dval]
                rdnT = [int(mval<dval),int(mval>=dval)]  # to use np.sum

            for i in 0,1:  # adds nesting per recursion
                DerT[i]+=[derT[i]]; ValT[i]+=[valT[i]]; RdnT[i]+=[rdnT[i]]

    return DerT,ValT,RdnT

def add_unpack(H, incr):  # recursive unpack hierarchy of unknown nesting to add input
    # new_H = []
    for i, e in enumerate(H):
        if isinstance(e,list):
            add_unpack(e,incr)
        else: H[i] += incr
    return H

def last_add(H, i):  # recursive unpack hierarchy of unknown nesting to add input
    while isinstance(H,list):
        H=H[-1]
    H+=i

def unpack(H):  # recursive unpack hierarchy of unknown nesting
    while isinstance(H,list):
        last_H = H
        H=H[-1]
    return last_H

def nest(P, ddepth=2):  # default ddepth is nest 2 times: tuple->layer->H, rngH is ptuple, derH is 1,2,4.. ptuples'layers?

    # fback adds alt fork per layer, may be empty?
    # agg+ adds depth: number brackets before the tested bracket: P.valT[0], P.valT[0][0], etc?

    if not isinstance(P.valT[0],list):
        curr_depth = 0
        while curr_depth < ddepth:
            P.derT[0]=[P.derT[0]]; P.valT[0]=[P.valT[0]]; P.rdnT[0]=[P.rdnT[0]]
            P.derT[1]=[P.derT[1]]; P.valT[1]=[P.valT[1]]; P.rdnT[1]=[P.rdnT[1]]
            curr_depth += 1

        if isinstance(P, CP):
            for derP in P.link_t[1]:
                curr_depth = 0
                while curr_depth < ddepth:
                    derP.derT[0]=[derP.derT[0]]; derP.valT[0]=[derP.valT[0]]; derP.rdnT[0]=[derP.rdnT[0]]
                    derP.derT[1]=[derP.derT[1]]; derP.valT[1]=[derP.valT[1]]; derP.rdnT[1]=[derP.rdnT[1]]
                    curr_depth += 1