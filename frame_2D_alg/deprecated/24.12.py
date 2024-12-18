# Link fork eval, projected by M of surrounding cluster:

vD = D * (M / ave)  # project by borrow from rel M
vM = M - vD / 2  # cancel by lend to D


def add_H(HE, He_, root=None, ri=None, sign=1):  # unpack derHs down to numericals and sum them

    if not isinstance(He_, list): He_ = [He_]
    for He in He_:
        if HE:
            for i, (Lay, lay) in enumerate(
                    zip_longest(HE.H, He.H, fillvalue=None)):  # cross comp layer
                if lay:
                    if Lay:
                        Lay.add_H(lay)
                    else:
                        if Lay is None:
                            HE.append_(lay.copy_(root=HE))  # pack a copy of new lay in HE.H
                        else:
                            HE.H[i] = lay.copy_(root=HE)  # Lay was []
            HE.add_lay(He, sign)
            HE.node_ += [node for node in He.node_ if
                         node not in HE.node_]  # node_ is empty in CL derH?
        elif root:
            if ri is None:  # root is node
                root.derH = He.copy_(root=root)
            else:
                root.H[ri] = He.copy_(root=root)
            root.Et += He.Et
        else:
            HE = He.copy_(root=root)

    return HE  # root should be updated by returned HE, we don't need it here?

def comp_N(_N,N, rn, angle=None, dist=None, dir=None):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = dir is not None  # compared links have binary relative direction
    dir = 1 if dir is None else dir  # convert to numeric
    # comp externals:
    if fd:
        _L, L = _N.dist, N.dist;  dL = _L-L; mL = min(_L,L) - ave_L  # direct match
        mA,dA = comp_angle(_N.angle, [d*dir for d in N.angle])  # rev 2nd link in llink
        # comp med if LL: isinstance(>nodet[0],CL), higher-order version of comp dist?
    else:
        _L, L = len(_N.node_),len(N.node_); dL = _L-L; mL = min(_L,L) - ave_L
        mA,dA = comp_area(_N.box, N.box)  # compare area in CG vs angle in CL
    n = .3
    M = mL + mA; D = (abs(dL)+abs(dA)) * (M/ave); M = M - D/2
    Et = np.array([M,D])
    md_t = [np.array([np.array([mL,mA]), np.array([dL,dA]), Et, n], dtype=object)]  # init as [mdExt]
    if not fd:  # CG
        mdLat = comp_latuple(_N.latuple,N.latuple, _N.n, N.n)  # not sure about ns
        mdLay = comp_md_(_N.mdLay[1], N.mdLay[1], rn, dir)
        md_t += [mdLat,mdLay]; Et += mdLat[2] + mdLay[2]; n += mdLat[3] + mdLay[3]
    # | n = (_n+n)/2?
    Et = np.append(Et, (_N.Et[2]+N.Et[2])/2 )  # Et[0] += ave_rn - rn?
    subLay = CH(n=n, md_t=md_t, Et=Et)
    eLay = CH(H=[subLay], n=n, md_t=deepcopy(md_t), Et=copy(Et))
    if _N.derH and N.derH:
        dderH = _N.derH.comp_H(N.derH, rn, dir=dir)  # comp shared layers
        eLay.append_(dderH, flat=1)
    elif _N.derH: eLay.append_(_N.derH.copy_(root=eLay))  # one empty derH
    elif  N.derH: eLay.append_(N.derH.copy_(root=eLay,rev=1))
    # spec: comp_node_(node_|link_), combinatorial, node_ may be nested with rng-)agg+, graph similarity search?
    Et = copy(eLay.Et)
    if not fd and _N.altG and N.altG:  # not for CL, eval M?
        altLink = comp_N(_N.altG, N.altG, _N.altG.n/N.altG.n)  # no angle,dist, init alternating PPds | dPs?
        eLay.altH = altLink.derH
        Et += eLay.altH.Et
    Link = CL(nodet=[_N,N],derH=eLay, n=min(_N.n,N.n),yx=np.add(_N.yx,N.yx)/2, angle=angle,dist=dist,box=extend_box(N.box,_N.box))
    if Et[0] > ave * Et[2]:
        eLay.root = Link
        for rev, node in zip((0,1), (N,_N)):  # reverse Link direction for _N
            if fd: node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            else:  node.rim += [(Link, rev)]
            node.extH.add_H(Link.derH, root=node.extH)
            # flat
    return Link

def copy_(_He, root, rev=0):
        # comp direction may be reversed
        He = CH(root=root, node_=copy(_He.node_), n=_He.n, i=_He.i, i_=copy(_He.i_))
        He.Et = copy(_He.Et)
        He.md_t = deepcopy(_He.md_t)
        if rev:
            for _,d_,_,_ in He.md_t:  # mdExt, possibly mdLat, mdLay
               d_ *= -1   # negate ds
        for he in _He.H:
            He.H += [he.copy_(root=He, rev=rev)] if he else [[]]
        return He

def comp_lay(_md_t, md_t, rn, dir=1):  # replace dir with rev?

    der_md_t = []
    tM, tD = 0, 0
    for i, (_md_, md_) in enumerate(zip(_md_t, md_t)):  # [mdExt, possibly mdLat, mdLay], default per layer
        if i == 2:  # nested mdLay
            sub_md_t, (m,d) = comp_lay(_md_[0], md_[0], rn, dir=dir)  # nested mdLay is [md_t, [tM, tD]], skip [tM and tD]
            der_md_t += [[sub_md_t, (m,d)]]
            tM += m; tD += d

    def comp_H(_He, He, dir=1):  # unpack each layer of CH down to numericals and compare each pair

        der_md_t, et = comp_md_t(_He.md_t, He.md_t, _He.n / He.n, dir=1)
        # ini:
        DLay = CH(md_t=der_md_t, Et=np.append(et,(_He.Et[2]+He.Et[2])/2), n=.3 if len(der_md_t)==1 else 2.3)  # .3 in default comp ext
        # empty H in bottom | deprecated layer:
        for rev, _lay, lay in zip((0,1), _He.H, He.H):  #  fork & layer CH / rng+|der+, flat
            if _lay and lay:
                dLay = _lay.comp_H(lay, dir)  # comp He.md_t, comp,unpack lay.H
                DLay.append_(dLay)  # DLay.H += subLay
            else:
                l = _lay if _lay else lay  # only one not empty lay = difference between lays:
                if l: DLay.append_(l.copy_(root=DLay, rev=rev))
                else: DLay.H += [CH()]  # to trace fork types
            # nested subHH ( subH?
        return DLay

    def comp_He(_He, He, rn, root, dir=1):

        derH = CH(root=root)
        # lay.H maps to higher Hs it was derived from, len lay.H = 2 ^ lay_depth (unpacked root H[:i])
        for _he, he in zip(_He.H, He.H):
            if he:
                if _he:  # maybe empty CH to trace fork types
                    if isinstance(he.H[0], CH):
                        Lay = _he.comp_He(he, rn, root=derH, dir=dir)  # deeper unpack -> comp_md_t
                    else:
                        Lay = _he.comp_md_t(he, rn=rn, root=derH, dir=dir)  # comp shared layers
                    derH.append_(Lay, flat=1)
                else:
                    derH.append_(he.copy_(root=derH, dir=1))  # not sure we need to copy?
        # old comp_H:
        for dir, _he, he in zip((1,-1), _He.H, He.H):  #  fork & layer CH / rng+|der+, flat
            if _he and he:
                Lay = _he.comp_He(he, root=derH, dir=dir)  # unpack lay.H
                derH.append_(Lay)
            else:
                l = _he if _he else he  # only one not empty lay = difference between lays:
                if l: derH.append_(l.copy_(root=derH, dir=dir))
                else: derH.H += [CH()]  # to trace fork types

        return derH

    def add_H(HE, He_, sign=1):  # unpack derHs down to numericals and sum them

        if not isinstance(He_,list): He_ = [He_]
        for He in He_:
            for i, (Lay,lay) in enumerate(zip_longest(HE.H, He.H, fillvalue=None)):  # cross comp layer
                if lay:
                    if Lay is None: HE.append_(lay.copy_(root=HE))  # pack a copy of new lay in HE.H
                    else:           Lay.add_H(lay)  # depth-first, He in add_lay has summed all the nested lays
            HE.add_lay(He, sign)
            HE.node_ += [node for node in He.node_ if node not in HE.node_]  # node_ is empty in CL derH?

        return HE  # root should be updated by returned HE

def trace_P_adjacency(edge):  # fill and trace across slices

    P_map_ = [(P, y,x) for P in edge.P_ for y,x in edge.rootd if edge.rootd[y,x] is P]
    prelink__ = defaultdict(list)  # uplinks
    while P_map_:
        _P, _y,_x = P_map_.pop(0)  # also pop _P__
        _margin = prelink__[_P]  # empty list per _P
        for y,x in [(_y-1,_x),(_y,_x+1),(_y+1,_x),(_y,_x-1)]:  # adjacent pixels
            try:  # form link if yx has _P
                P = edge.rootd[y,x]
                margin = prelink__[P]  # empty list per P
                if _P is not P:
                    if _P.yx < P.yx and _P not in margin:
                        margin += [_P]  # _P is higher
                    elif P not in _margin:
                        _margin += [P]  # P is higher
            except KeyError:  # if yx empty, keep tracing
                if (y,x) not in edge.dert_: continue   # yx is outside the edge
                edge.rootd[y,x] = _P
                P_map_ += [(_P, y,x)]
    # remove crossed links
    for P in edge.P_:
        yx = P.yx
        for _P, __P in combinations(prelink__[P], r=2):
            _yx, __yx = _P.yx, __P.yx
            # get aligned line segments:
            _yx1 = np.subtract(_P.yx_[0], _P.axis)
            _yx2 = np.add(_P.yx_[-1], _P.axis)
            __yx1 = np.subtract(__P.yx_[0], __P.axis)
            __yx2 = np.add(__P.yx_[-1], __P.axis)
            # remove crossed uplinks:
            if xsegs(yx, _yx, __yx1, __yx2):
                prelink__[P].remove(_P)
            elif xsegs(yx, __yx, _yx1, _yx2):
                prelink__[P].remove(__P)
    # for comp_slice:
    edge.pre__ = prelink__

def match_signed(_par, par): # for signed pars

    match = min(abs(_par),abs(par))
    return -match if (_par<0) != (par<0) else match    # match = neg min if opposite-sign comparands

class CP(CBase):

    def __init__(P, edge, yx, axis):
        super().__init__()
        y, x = yx
        P.axis = ay, ax = axis
        pivot = i,gy,gx,g = edge.dert_[y,x]  # dert is None if _y,_x not in edge.dert_: return` in `interpolate2dert`
        ma = ave_dangle  # max if P direction = dert g direction
        m = ave_g - g
        pivot += ma,m
        edge.rootd[y, x] = P
        I,G,M,Ma,L,Dy,Dx = i,g,m,ma,1,gy,gx
        P.yx_, P.dert_ = [yx], [pivot]

        for dy,dx in [(-ay,-ax),(ay,ax)]:  # scan in 2 opposite directions to add derts to P
            P.yx_.reverse(); P.dert_.reverse()
            (_y,_x), (_,_gy,_gx,*_) = yx, pivot  # start from pivot
            y,x = _y+dy, _x+dx  # 1st extension
            while True:
                # scan to blob boundary or angle miss:
                ky, kx = round(y), round(x)
                if (round(y),round(x)) not in edge.dert_: break
                try: i,gy,gx,g = interpolate2dert(edge, y, x)
                except TypeError: break  # out of bound (TypeError: cannot unpack None)
                if edge.rootd.get((ky,kx)) is not None: break  # skip overlapping P
                mangle, dangle = comp_angle((_gy,_gx), (gy, gx))
                if mangle < ave_dangle: break  # terminate P if angle miss
                # update P:
                edge.rootd[ky, kx] = P
                m = ave_g - g
                I += i; Dy += dy; Dx += dx; G += g; Ma += ma; M += m; L += 1
                P.yx_ += [(y,x)]; P.dert_ += [(i,gy,gx,g,ma,m)]
                # for next loop:
                y += dy; x += dx
                _y,_x,_gy,_gx = y,x,gy,gx

        P.yx = tuple(np.mean([P.yx_[0], P.yx_[-1]], axis=0))
        P.latuple = new_latuple(I, G, M, Ma, L, [Dy, Dx])

def agg_cluster_(frame):  # breadth-first (node_,L_) cross-comp, clustering, recursion

    def cluster_eval(G, N_, fd):
        pL_ = {l for n in N_ for l,_ in get_rim(n, fd)}
        if len(pL_) > ave_L:
            sG_ = cluster_N_(G, pL_, fd)  # optionally divisive clustering
            frame.subG_ = sG_
            for sG in sG_:
                if len(sG.subG_) > ave_L:
                    find_centroids(sG)  # centroid clustering in sG.node_ or subG_?
    '''
    cross-comp G_) GG_) GGG_., interlaced with exemplar centroid selection 
    '''
    N_,L_,Et = comp_node_(frame.subG_)  # cross-comp exemplars, extrapolate to their node_s?
    if val_(Et,fo=1) > 0:
        fd = 0
        mlay = CH().add_H([L.derH for L in L_])  # mfork, else no new layer
        frame.derH = CH(H=[mlay], root=frame, Et=copy(mlay.Et)); mlay.root=frame.derH
        if val_(Et,mEt=Et) > 0:  # same root
            for L in L_:
                L.extH, L.root, L.mL_t, L.rimt, L.aRad, L.visited_, L.Et = CH(), frame, [[],[]], [[],[]], 0, [L], copy(L.derH.Et)
            lN_,lL_,dEt = comp_link_(L_,Et)  # comp new L_, root.link_ was compared in root-forming for alt clustering
            if val_(dEt, mEt=Et, fo=1) > 0:  # recursive der+ eval_: cost > ave_match, add by feedback if < _match?
                fd = 1
                frame.derH.append_(CH().add_H([L.derH for L in lL_]))  # dfork
        else:
            frame.derH.H += [[]]  # empty to decode rng+|der+, n forks per layer = 2^depth
        # + aggLays, derLays, exemplars:
        cluster_eval(frame, N_, fd=0)
        if fd:
            cluster_eval(frame, lN_, fd=1)

    def comp_H(_He, He, rn, root):

        derH = CH(root=root)  # derH.H ( flat lay.H, or nest per higher lay.H for selective access?
        # lay.H maps to higher Hs it was derived from, len lay.H = 2 ^ lay_depth (unpacked root H[:i])

        for _lay, lay in zip_longest(_He.H, He.H, fillvalue=None):  # both may be empty CH to trace fork types
            if _lay:
                if lay:
                    if isinstance(lay.H[0], CH):  # same depth in _lay
                        dLay = _lay.comp_H(lay, rn, root=derH)  # deeper unpack -> comp_md_t
                    else:
                        dLay = _lay.comp_md_C(lay, rn=rn, root=derH, olp=(_He.Et[3]+He.Et[3]) /2)  # comp shared layers, add n to olp?
                    derH.append_(dLay)
                else:
                    derH.append_(_lay.copy_(root=derH, dir=1))  # diff = _he
            elif lay:
                derH.append_(lay.copy_(root=derH, dir=-1))  # diff = -he
            # no fill if both empty?
        return derH
