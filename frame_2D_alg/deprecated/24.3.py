def der_recursion(root, PP, fd=0):  # node-mediated correlation clustering: keep same Ps and links, increment link derH, then P derH in sum2PP

    if fd:  # add prelinks per P if not initial call:
        for P in PP.P_: P.link_ += [copy(unpack_last_link_(P.link_))]

    rng_recursion(PP, rng=1, fd=fd)  # extend PP.link_, derHs by same-der rng+ comp

    form_PP_t(PP, PP.P_, iRt = PP.Et[2:4] if PP.Et else [0,0])  # der+ is mediated by form_PP_t
    if root: root.fback_ += [[PP.He, PP.et]]  # feedback from PPds


def rng_recursion(PP, rng=1, fd=0):  # similar to agg+ rng_recursion, but contiguously link mediated, because

    iP_ = PP.P_
    while True:
        P_ = []; V = 0
        for P in iP_:
            if not P.link_: continue
            prelink_ = []  # new prelinks per P
            _prelink_ = P.link_.pop()  # old prelinks per P
            for _link in _prelink_:
                _P = _link._P if fd else _link
                dy,dx = np.subtract(_P.yx, P.yx)
                distance = np.hypot(dy,dx)  # distance between P midpoints, /= L for eval?
                if distance < rng:  # | rng * ((P.val+_P.val) / ave_rval)?
                    mlink = comp_P(_link if fd else [_P,P, distance,[dy,dx]], fd)  # return link if match
                    if mlink:
                        V += mlink.et[0]  # unpack last link layer:
                        link_ = P.link_[-1] if P.link_ and isinstance(P.link_[-1], list) else P.link_  # der++ if PP.He[0] depth==1
                        if rng > 1:
                            if rng == 2: link_[:] = [link_[:]]  # link_ -> link_H
                            if len(link_) < rng: link_ += [[]]  # new link_
                            link_ = link_[-1]  # last rng layer
                        link_ += [mlink]
                        prelink_ += unpack_last_link_(_P.link_[:-1])  # get last link layer, skip old prelinks
            if prelink_:
                if not fd: prelink_ = [link._P for link in prelink_]  # prelinks are __Ps, else __links
                P.link_ += [prelink_]  # temporary prelinks
                P_ += [P]  # for next loop
        rng += 1
        if V > ave * len(P_) * 6:  #  implied val of all __P_s, 6: len mtuple
            iP_ = P_
        else:
            for P in P_: P.link_.pop()
            break
    PP.rng=rng
    '''
    der++ is tested in PPds formed by rng++, no der++ inside rng++: high diff @ rng++ termination only?
    '''

def comp_P(link, fd):

    if isinstance(link,z): _P, P = link._P, link.P  # in der+
    else:                  _P, P, S, A = link  # list in rng+
    rn = len(_P.dert_) / len(P.dert_)

    if _P.He and P.He:
        # der+: append link derH, init in rng++ from form_PP_t
        depth,(vm,vd,rm,rd),H, n = comp_(_P.He, P.He, rn=rn)
        rm += vd > vm; rd += vm >= vd
        aveP = P_aves[1]
    else:
        # rng+: add link derH
        H = comp_ptuple(_P.ptuple, P.ptuple, rn)
        vm = sum(H[::2]); vd = sum(abs(d) for d in H[1::2])
        rm = 1 + vd > vm; rd = 1 + vm >= vd
        aveP = P_aves[0]
        n = 1  # 6 compared params is a unit of n

    if vm > aveP*rm:  # always rng+
        if fd:
            He = link.He
            if not He[0]: He = link.He = [1,[*He[1]],[He]]  # nest md_ as derH
            He[1] = np.add(He[1],[vm,vd,rm,rd])
            He[2] += [[0, [vm,vd,rm,rd], H]]  # nesting, Et, H
            link.et = [V+v for V, v in zip(link.et,[vm,vd, rm,rd])]
        else:
            link = CderP(P=P,_P=_P, He=[0,[vm,vd,rm,rd],H], et=[vm,vd,rm,rd], S=S, A=A, n=n, roott=[[],[]])

        return link


def form_PP_t(root, P_, iRt):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    for fd in 0,1:
        P_Ps = []; Link_ = []
        for P in P_:  # not PP.link_: P uplinks are unique, only G links overlap
            Ps = []
            for derP in unpack_last_link_(P.link_):
                Ps += [derP._P]; Link_ += [derP]  # not needed for PPs?
            P_Ps += [Ps]  # aligned with P_
        inP_ = []  # clustered Ps and their val,rdn s for all Ps
        for P in root.P_:
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

    for PP in PP_t[1]:  # eval der+ / PPd only, after form_PP_t -> P.root
        if PP.Et[1] * len(PP.link_) > PP_aves[1] * PP.Et[3]:
            # node-mediated correlation clustering:
            der_recursion(root, PP, fd=1)
        if root.fback_:
            feedback(root)  # after der+ in all nodes, no single node feedback

    root.node_ = PP_t  # nested in der+, add_alt_PPs_?


def sum2PP(root, P_, derP_, iRt, fd):  # sum links in Ps and Ps in PP

    PP = CPP(typ='PP',fd=fd,root=root,P_=P_,rng=root.rng+1, Et=[0,0,1,1], et=[0,0,1,1], link_=[], box=[0,0,0,0],  # not inf,inf,-inf,-inf?
           ptuple = z(typ='ptuple',I=0, G=0, M=0, Ma=0, angle=[0,0], L=0), He=[])
    # += uplinks:
    S,A = 0, [0,0]
    for derP in derP_:
        if derP.P not in P_ or derP._P not in P_: continue
        if derP.He:
            add_(derP.P.He, derP.He, iRt)
            add_(derP._P.He, negate(deepcopy(derP.He)), iRt)
        PP.link_ += [derP]; derP.roott[fd] = PP
        PP.Et = [V+v for V,v in zip(PP.Et, derP.et)]
        PP.Et[2:4] = [R+ir for R,ir in zip(PP.Et[2:4], iRt)]
        derP.A = np.add(A,derP.A); S += derP.S
    PP.ext = [len(P_), S if S != 0 else 1, A]  # all from links  (prevent zero S for single P's PP)

    # += Ps:
    celly_,cellx_ = [],[]
    for P in P_:
        PP.area += P.ptuple.L
        PP.ptuple += P.ptuple
        if P.He:
            add_(PP.He, P.He)
            PP.et = [V+v for V, v in zip(PP.et, P.He[1])]  # we need to sum et from P too? Else they are always empty
        for y,x in P.cells:
            PP.box = accum_box(PP.box, y, x); celly_+=[y]; cellx_+=[x]
    # pixmap:
    y0,x0,yn,xn = PP.box
    PP.mask__ = np.zeros((yn-y0, xn-x0), bool)
    celly_ = np.array(celly_); cellx_ = np.array(cellx_)
    PP.mask__[(celly_-y0, cellx_-x0)] = True

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

    _I, _G, _M, _Ma, (_Dy, _Dx), _L = _ptuple.I, _ptuple.G, _ptuple.M,_ptuple.Ma,_ptuple.angle,_ptuple.L
    I, G, M, Ma, (Dy, Dx), L = ptuple.I, ptuple.G, ptuple.M,ptuple.Ma,ptuple.angle,ptuple.L

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

    while link_ and isinstance(link_[-1], list): link_ = link_[-1]
    return link_

def nest(HE,He):

    Depth, depth = HE[0], He[0]  # nesting depth, nest to the deeper He: md_-> derH-> subH-> aggH:
    ddepth = abs(Depth - depth)
    if ddepth:
        nHe = [HE,He][Depth > depth]  # nested He
        while ddepth > 0:
           nHe[:] = [nHe[0]+1, [*nHe[1]], [deepcopy(nHe)]]; ddepth -= 1

def node_connect(_G_):  # node connectivity = sum surround link vals, incr.mediated: Graph Convolution of Correlations
    '''
    Aggregate direct * indirect connectivity per node from indirect links via associated nodes, in multiple cycles.
    Each cycle adds contributions of previous cycles to linked-nodes connectivity, propagated through the network.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png
    '''
    while True:
        # eval accumulated G connectivity vals, indirectly extending their range
        G_ = []  # next connectivity expansion, more selective by DVt,Lent = [0,0],[0,0]?
        for G in _G_:
            uprim = []  # >ave updates of direct links
            rim = G.rim_H[-1] if G.rim_H and isinstance(G.rim_H[0], list) else G.rim_H
            for i in 0,1:
                val,rdn,dec = G.Et[i::2]  # connect by last layer
                ave = G_aves[i]
                for link in rim:
                    # >ave derG in fd rim
                    lval,lrdn,ldec = link.Et[i::2]; ldec /= link.n
                    _G = link._node if link.node is G else link.node
                    _val,_rdn,_dec = _G.Et[i::2]
                    # Vt.. for segment_node_:
                    V = ldec * (val+_val); dv = V-lval
                    R = ldec * (rdn+_rdn); dr = R-lrdn
                    D = ldec * (dec+_dec); dd = D-ldec
                    link.Et[i::2] = [V, R, D]
                    if dv > ave * dr:
                        G.Et[i::2] = [V+v for V,v in zip(G.Et[i::2], [V, R, D])]  # add link last layer vals
                        if link not in uprim: uprim += [link]
                        # more selective eval: dVt[i] += dv; L=len(uprim); Lent[i] += L
                    if V > ave * R:
                        G.et[i::2] = [V+v for V, v in zip(G.et[i::2], [dv, dr, dd])]
            if uprim:  # prune rim for next loop
                rim[:] = uprim
                G_ += [G]
        if G_: _G_ = G_  # exclude weakly incremented Gs from next connectivity expansion loop
        else:  break

def segment_node_(root, root_G_, fd, nrng, fagg):  # eval rim links with summed surround vals for density-based clustering

    # graph += [node] if >ave (surround connectivity * relative value of link to any internal node)
    igraph_ = []; ave = G_aves[fd]

    for G in root_G_:   # init per node, last-layer Vt,Vt,Dt:
        grapht = [[G],[],copy(G.Et), copy(G.rim_H[-1] if G.rim_H and isinstance(G.rim_H[0],list) else G.rim_H)]  # link_ = last rim
        G.root = grapht  # for G merge
        igraph_ += [grapht]
    _graph_ = igraph_

    while True:
        graph_ = []
        for grapht in _graph_:  # extend grapht Rim with +ve in-root links
            G_, Link_, Et, Rim = grapht
            inVal, inRdn = 0,0  # new in-graph +ve
            new_Rim = []
            for link in Rim:  # unique links
                if link.node in G_:
                    G = link.node; _G = link._node
                else:
                    G = link._node; _G = link.node
                if _G in G_: continue
                # connect by rel match of nodes * match of node Vs: surround M|Ds,
                # V suggests how deeply inside the graph is G
                cval = link.Et[fd] + get_match(G.Et[fd],_G.Et[fd])  # same coef for int and ext match?
                crdn = link.Et[2+fd] + (G.Et[2+fd] + _G.Et[2+fd]) / 2
                if cval > ave * crdn:  # _G and its root are effectively connected
                    # merge _root:
                    _grapht = _G.root
                    _G_,_Link_,_Et,_Rim = _grapht
                    if link not in Link_: Link_ += [link]
                    Link_[:] += [_link for _link in _Link_ if _link not in Link_]
                    for g in _G_:
                        g.root = grapht
                        if g not in G_: G_+=[g]
                    for i in 0,1:
                        Et[:] = [V+v for V,v in zip(Et, _Et)]
                        inVal += _Et[fd]; inRdn += _Et[2+fd]
                    if _grapht in igraph_:
                        igraph_.remove(_grapht)
                    new_Rim += [link for link in _Rim if link not in new_Rim+Rim+Link_]
            # for next loop:
            if len(new_Rim) * inVal > ave * inRdn:
                graph_ += [[G_,Link_, Et, new_Rim]]

        if graph_: _graph_ = graph_  # selected graph expansion
        else: break

    graph_ = []
    for grapht in igraph_:
        if grapht[2][fd] > ave * grapht[2][2+fd]:  # form Cgraphs if Val > ave* Rdn
            graph_ += [sum2graph(root, grapht[:3], fd, nrng, fagg)]

    return graph_


def sum2graph(root, grapht, fd, nrng, fagg):  # sum node and link params into graph, aggH in agg+ or player in sub+

    G_,Link_, Et = grapht

    graph = CG(fd=fd, node_=G_,link_=Link_, et=Et, rng=nrng)
    # Et and et are not needed, same for latuple=[0,0,0,0,0, [0,0]], n=1?
    if fd: graph.root = root
    et = [0,0,0,0,0,0]  # current vals
    for link in Link_:  # unique current-layer links
        link.root = graph; graph.S += link.S; np.add(graph.A,link.A)
        et = [V+v for V,v in zip(et, link.Et)]
        graph.n += link.n
    eDerH, Et = [],[0,0,0,0,0,0]
    # grapht int = node int+ext
    for G in G_:
        graph.area += G.area
        sum_last_lay(G, fd)
        graph.box = extend_box(graph.box, G.box)
        graph.latuple = [P+p for P,p in zip(graph.latuple[:-1],graph.latuple[:-1])] + [[A+a for A,a in zip(graph.latuple[-1],graph.latuple[-1])]]
        add_(graph.derH, G.derH,irdnt=[1,1])
        eDerH = add_(eDerH, G.ederH, irdnt=G.et[2:4])
        graph.aggH = add_(graph.aggH, G.aggH)
        Et = [V+v for V,v in zip(Et, G.Et)]  # combined across node layers
    # dsubH| daggH:
    if graph.aggH: graph.aggH[2] += [eDerH]  # init []
    else: graph.aggH = eDerH
    graph.Et = [V+v+ev for V,v,ev in zip(graph.Et, Et, et)]  # graph internals = G Internals + Externals, combined across link layers
    if fagg:
        if graph.aggH and graph.aggH[0] == 1:  # if depth == 1 (derH), convert to subH (depth =2)
            # 1st agg+, init aggH = [subHv]:
            graph.aggH =  [2, [*graph.et], graph.aggH]

    if fd:  # assign alt graphs from d graph, after both linked m and d graphs are formed
        for link in graph.link_:
            mgraph = link.roott[0]
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]
                        G.alt_Et = [V+v for V,v in zip_longest(G.alt_Et, alt_G.et, fillvalue=0)]
    return graph

def comp_G(link, iEt):  # add flat dderH to link and link to the rims of comparands

    _G, G = link._node, link.node
    rn = len(_G.node_)/ len(G.node_)  # or unpack nested node_?
    Depth = 0
    # / P, default
    Et, md_ = comp_latuple(_G.latuple, G.latuple, rn, fagg=1)
    dderH_H = [[0,[*Et], md_]]  # dderH.H
    N = 1
    # / PP, if >1 Ps:
    if _G.derH and G.derH:
        depth, et, dH, n = comp_(_G.derH, G.derH, rn, fagg=1)  # generic dderH
        dderH_H += [[depth, et, dH]]  # add flat
        Et = [E+e for E,e in zip(Et,et)]  # evaluation tuple: valt, rdnt, dect
        N += n; Depth = max(1,depth)
    # / G, if >1 PPs:
    if _G.aggH and G.aggH:  # exactly as above?
        depth, et, daggH, n = comp_(_G.aggH, G.aggH, rn, fagg=1)
        dderH_H += daggH  # still flat
        Et = [E+e for E,e in zip(Et,et)]
        N += n; Depth = depth
    # if not single-node:
    if _G.S and G.S:
        et, extt = comp_ext((len(_G.node_),_G.S,_G.A),(len(G.node_),G.S,G.A), rn)  # or unpack?
        Et = [E+e for E,e in zip(Et,et)]; N += .5
        dderH_H += [[0, et, extt]]  # must be last
    else:
        dderH_H += [[]]
        # for fixed len layer to decode nesting, else use Cext as a terminator?
    dderH = [Depth, Et, dderH_H]
    link.Et = Et; link.n = N  # reset per comp_G
    if link.dderH:
        add_(link.dderH, dderH)  # existing dderH in der+
    else:
        link.dderH = dderH
    for fd in 0, 1:
        Val, Rdn, Dec = Et[fd::2]
        if Val > G_aves[fd] * Rdn:
            iEt[fd::2] = [V+v for V,v in zip(iEt[fd::2], Et[fd::2])]  # to eval grapht in form_graph_t
            if not fd:
                for G in link.node, link._node:
                    rim_H = G.rim_H
                    if rim_H and isinstance(rim_H[0],list):  # rim is converted to rim_H in 1st sub+
                        if len(rim_H) == len(G.Rim_H): rim_H += [[]]  # no new rim layer yet
                        rim_H[-1] += [link]  # rim_H
                    else:
                        rim_H += [link]  # rim
