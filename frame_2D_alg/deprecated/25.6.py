def sum2graph(root, node_,link_,llink_,Et,olp, Lay, rng, fi):  # sum node and link params into graph, aggH in agg+ or player in sub+
    '''
            N = sum_N_(node_ if fnode_ else link_, fCG = fnode_)
            m_,M = centroid_M(N.derTT[0],ave*nrc)  # weigh by match to mean m|d
            d_,D = centroid_M(N.derTT[1],ave*nrc); derTT = np.array([m_,d_])
            Et += np.array([M, D, Et[2]]) * int_w
    '''
    n0 = node_[0]
    graph = CG( fi=fi,rng=rng,olp=olp, Et=Et,et=Lay.Et, N_=node_,L_=link_, root=root,
                box=n0.box, baseT=Lay.baseT+n0.baseT, derTT=Lay.derTT+n0.derTT, derH=copy(n0.derH))
    graph.hL_ = llink_
    n0.root = graph; yx_ = [n0.yx]; fg = fi and isinstance(n0.N_[0],CG)  # not PPs
    Nt = copy_(n0)  #->CN, comb forks: add_N(Nt,Nt.Lt)?
    for N in node_[1:]:
        add_H(graph.derH,N.derH,graph); graph.baseT+=N.baseT; graph.derTT+=N.derTT; graph.box=extend_box(graph.box,N.box); yx_+=[N.yx]
        if fg: add_N(Nt, N)
    if fg: graph.H = Nt.H + [Nt]  # pack prior top level
    graph.derH += [[Lay,[]]]  # append flat
    yx = np.mean(yx_,axis=0); dy_,dx_ = (yx_-yx).T; dist_ = np.hypot(dy_,dx_)
    graph.span = dist_.mean()  # node centers distance to graph center
    graph.angle = np.sum([l.angle for l in link_],axis=0)
    graph.yx = yx
    if not fi:  # add mfork as link.node_(CL).root dfork, 1st layer, higher layers are added in cross_comp
        for L in link_:  # LL from comp_link_
            LR_ = set([n.root for n in L.N_ if isinstance(n.root,CG)])  # nodet, skip frame, empty roots
            if LR_:
                dfork = reduce(lambda F,f: F.add_lay(f), L.derH, CLay())  # combine lL.derH
                for LR in LR_:
                    LR.Et += dfork.Et; LR.derTT += dfork.derTT  # lay0 += dfork
                    if LR.derH[-1][1]: LR.derH[-1][1].add_lay(dfork)  # direct root only
                    else:              LR.derH[-1][1] =dfork.copy_()  # was init by another node
                    if LR.lH: LR.lH[-1].N_ += [graph]  # last lev
                    else:     LR.lH += [CN(N_=[graph])]  # init
        alt_=[]  # add mGs overlapping dG
        for L in node_:
            for n in L.N_:  # map root mG
                mG = n.root
                if isinstance(mG, CG) and mG not in alt_:  # root is not frame
                    mG.alt_ += [graph]  # cross-comp|sum complete alt before next agg+ cross-comp, multi-layered?
                    alt_ += [mG]
    return graph
'''
    else:
        # add mfork as link.node_(CL).root dfork, 1st layer, higher layers are added in cross_comp
        for L in link_:  # LL from comp_link_
            LR_ = set([n.root for n in L.N_ if isinstance(n.root,CN)])  # nodet, skip frame, empty roots
            if LR_:
                dfork = sum_H(L.derH)  # combine lL.derH into single CLay
                for LR in LR_:
                    LR.Et += dfork.Et; LR.derTT += dfork.derTT  # lay0 += dfork
                    if LR.derH[-1][1]: LR.derH[-1][1].add_lay(dfork)  # direct root only
                    else:              LR.derH[-1][1] = dfork.copy_()  # was init by another node
                    if LR.lH: LR.lH[-1].N_ += [graph]  # last lev
                    else:     LR.lH += [CN(N_=[graph])]  # init


            n_ = node_ if fnode_ else link_
            n0 = n_[0]; derH, derTT, baseT, Et, olp = copy_(n0.derH), n0.derTT.copy, n0.baseT.copy, n0.Et.copy, n0.olp
            for n in n_[1:]:
                add_H(derH,n.derH); derTT += n.derTT; baseT += n.baseT; Et += n.Et; olp+=n.olp
'''

def comp_H(H,h, rn, root, fi):  # one-fork derH if not fi, else two-fork derH

    derH, derTT, Et = [], np.zeros((2,8)), np.zeros(3)
    for _lay,lay in zip_longest(H,h):  # different len if lay-selective comp
        if _lay and lay:
            if fi:  # two-fork lays
                dLay = []
                for _fork,fork in zip_longest(_lay,lay):
                    if _fork and fork:
                        dlay = _fork.comp_lay(fork, rn,root=root)
                        if dLay: dLay.add_lay(dlay)  # sum ds between input forks
                        else:    dLay = dlay
            else:  # one-fork lays
                 dLay = _lay.comp_lay(lay, rn, root=root)
            derTT = dLay.derTT; Et += dLay.Et
            derH += [dLay]
    return derH, derTT, Et

def comp_N(_N,N, ave, fi, angle=None, span=None, dir=1, fdeep=0, rng=1):  # compare links, relative N direction = 1|-1, no need for angle, dist?

    dderH = []
    derTT, Et, rn = base_comp(_N, N, dir)
    # link M,D,A:
    baseT = np.array([*(_N.baseT[:2] + N.baseT[:2]*rn) / 2, *angle])  # redundant angle for generic base_comp, also span-> density?
    _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
    o = (_N.olp+N.olp) / 2
    Link = CN(rng=rng, olp=o, N_=[_N,N], baseT=baseT, derTT=derTT, yx=np.add(_N.yx,N.yx)/2, span=span, angle=angle, box=box)
    # spec / lay:
    if fdeep and (val_(Et, mw=len(N.derH)-2, aw=o) > 0 or N.name=='link'):  # else derH is dext,vert
        dderH = comp_H(_N.derH, N.derH, rn, Link, Et, fi)  # comp shared layers, if any
        # comp int proj x ext ders:
        # comp_H( proj_dH(_N.derH[1:]), dderH[:-1])  why 1: and :-1?
        # spec/ comp_node_(node_|link_)
    Link.derH = [CLay(root=Link, Et=Et, node_=[_N,N],link_=[Link], derTT=copy(derTT)), *dderH]
    for lay in dderH:
        derTT += lay.derTT; Et += lay.Et
    # spec / alt:
    if fi and _N.alt_ and N.alt_:
        et = _N.alt_.Et + N.alt_.Et  # comb val
        if val_(et, aw=2+o, fi=0) > 0:  # eval Ds
            Link.alt_L = comp_N(N2G(_N.alt_), N2G(N.alt_), ave*2, fi=1, angle=angle); Et += Link.alt_L.Et
    Link.Et = Et
    if Et[0] > ave * Et[2]:
        for rev, node, _node in zip((0,1),(N,_N),(_N,N)):  # reverse Link dir in _N.rimt
            node.et += Et
            node.nrim += [node]
            if fi: node.rim += [(Link,rev)]
            else: node.rim[1-rev] += [(Link,rev)]  # rimt opposite to _N,N dir
            add_H(node.extH, Link.derH, root=node, rev=rev, fi=0)
    return Link

def add_H(H, h, root, rev=0, fi=1):  # add fork L.derHs

    for Lay,lay in zip_longest(H,h):  # different len if lay-selective comp
        if lay:
            if fi:  # two-fork lays
                if Lay:
                    for Fork,fork in zip_longest(Lay,lay):
                        if fork:
                            if Fork: Fork.add_lay(fork,rev)
                            else:    Lay+= [fork.copy_(rev)]
                else:
                    Lay = []
                    for fork in lay:
                        if fork: Lay += [fork.copy_(rev)]; root.derTT += fork.derTT; root.Et += fork.Et
                        else:    Lay += [[]]
                    H += [Lay]
            else:  # one-fork lays
                if Lay: Lay.add_lay(lay,rev)
                else:   H += [lay.copy_(rev)]

def cross_comp(iN_, rc, root, fi=1):  # rc: redundancy+olp; (cross-comp, exemplar selection, clustering), recursion

    N__,L_,Et = comp_node_(iN_,rc) if fi else comp_link_(iN_,rc)  # CLs if not fi, olp comp_node_(C_)?
    if N__:
        Nt, n__ = [],[]
        for n in {N for N_ in N__ for N in N_}: n__ += [n]; n.sel = 0  # for cluster_N_
        # mfork:
        if val_(Et, mw=(len(n__)-1)*Lw, aw=rc+loop_w) > 0:
            E_,eEt = select_exemplars(root, n__, rc+loop_w, fi)  # typical nodes, refine by cluster_C_
            if val_(eEt, mw=(len(E_)-1)*Lw, aw=rc+clust_w) > 0:
                for rng, N_ in enumerate(N__,start=1):  # bottom-up rng incr
                    rng_E_ = [n for n in N_ if n.sel]   # cluster via rng exemplars
                    if rng_E_ and val_(np.sum([n.Et for n in rng_E_],axis=0), mw=(len(rng_E_)-1)*Lw, aw=rc+clust_w*rng) > 0:
                        hNt = cluster_N_(rng_E_, rc+clust_w*rng, fi,rng)
                        if hNt: Nt = hNt  # else keep lower rng
                if Nt and val_(Nt.Et, Et, mw=(len(Nt.N_)-1)*Lw, aw=rc+clust_w*rng+loop_w) > 0:
                    # cross_comp root.C_| mix in exclusive N_?
                    cross_comp(Nt.N_, rc+clust_w*rng+loop_w, root=Nt)  # top rng, select lower-rng spec comp_N: scope+?
        # dfork
        root.L_ = L_  # top N_ links
        mLay = CLay(); [mLay.add_lay(lay) for L in L_ for lay in L.derH]  # comb,sum L.derHs
        root.angle = np.sum([L.angle for L in L_],axis=0)
        LL_, Lt, dLay = [], [], []
        dval = val_(Et, mw=(len(L_)-1)*Lw, aw=rc+3+clust_w, fi=0)
        if dval > 0:
            L_ = N2G(L_)
            if dval > ave:  # recursive derivation -> lH/nLev, rng-banded?
                LL_ = cross_comp(L_, rc+loop_w*2, root, fi=0)  # comp_link_, no centroids?
                if LL_: dLay = CLay(); [dLay.add_lay(lay) for LL in LL_ for lay in LL.derH]  # optional dfork
            else:  # lower res
                Lt = cluster_N_(L_, rc+clust_w*2, fi=0, fnode_=1)
                if Lt: root.et += Lt.Et; root.lH += [Lt] + Lt.H  # link graphs, flatten H if recursive?
        root.derH += [[mLay,dLay]]
        if Nt:
            # feedback:
            lev = CN(N_=Nt.N_, Et=Nt.Et)  # N_ is new top H level
            add_NH(root.H, Nt.H+[lev], root)  # same depth?
            if Nt.H:  # lower levs: derH,H if recursion
                root.N_ = Nt.H.pop().N_  # top lev nodes
            comb_alt_(Nt.N_, rc + clust_w * 3)  # from dLs

    if not fi: return L_  # LL_

class CG(CN):  # PP | graph | blob: params of single-fork node_ cluster
    # graph / node
    name = "graph"
    def __init__(g, **kwargs):
        super().__init__(**kwargs)
        g.rim   = kwargs.get('rim', [])  # external links, or nodet if not fi?
        g.nrim  = kwargs.get('nrim',[])  # rim-linked nodes, = rim if not fi
        g.baset = kwargs.get('baset',np.zeros(4))  # sum baseT from rims
        g.extH  = kwargs.get('extH',[])  # sum derH from rims, single-fork
        g.extTT = kwargs.get('extTT',np.zeros((2,8)))  # sum from extH
    def __bool__(g): return bool(g.N_)  # never empty

N_pars = ['N_', 'L_', 'Et', 'et', 'H', 'lH', 'C_', 'fi', 'olp', 'derH','rng', 'baseT', 'derTT', 'yx', 'box', 'span', 'angle', 'root']
G_pars = ['extH', 'extTT', 'rim', 'nrim', 'alt_', 'fin', 'hL_'] + N_pars

def copy_(N, root=None, fCG=0, init=0):

    C, pars = (CG(),G_pars) if fCG else (CN(),N_pars)
    C.root = root
    for name, val in N.__dict__.items():
        if name == "_id" or name not in pars: continue  # for param pruning, not really used now
        elif name == "N_" and init: C.N_ = [N]
        elif name == "H" and init: C.H = []
        elif name == 'derH':
            for lay in N.derH:
                C.derH += [[fork.copy_() if fork else [] for fork in lay]] if isinstance(lay,list) else [lay.copy_()]  # single-fork
        elif name == 'extH': C.extH = [lay.copy_() for lay in N.extH]  # single fork
        elif isinstance(val,list) or isinstance(val,np.ndarray):
            setattr(C, name, copy(val))  # Et,yx,box, node_,link_,rim, alt_, baseT, derTT
        else:
            setattr(C, name, val)  # numeric or object: fi, root, span, nnest, lnest
    return C

def project_N_(Fg, y, x):

    angle = np.zeros(2); iDist = 0
    for L in Fg.L_:
        span = L.span; iDist += span; angle += L.angle / span  # unit step vector
    _dy,_dx = angle / np.hypot(*angle)
    dy, dx  = Fg.yx - np.array([y,x])
    foc_dist = np.hypot(dy,dx)
    rel_dist = foc_dist / (iDist / len(Fg.L_))
    cos_dang = (dx*_dx + dy*_dy) / foc_dist
    proj = cos_dang * rel_dist  # dir projection
    ET,eT = np.zeros((2,3)); DerTT,RimTT = np.zeros(((2,2,8)))
    N_ = []
    for _N in Fg.N_:
        # sum _N-specific projections for cross_comp
        (M,D,n),(m,d,en), I,eI = _N.Et,_N.rim.Et, _N.baseT[0],_N.baset[0]
        rn = en/n
        dec = rel_dist * ((M+m*rn) / (I+eI*rn))  # match decay rate, same for ds, * ddecay?
        derH = proj_dH(_N.derH,proj,dec);     derTT, Et = val_H(derH)
        rimH = proj_dH(_N.rim.derH,proj,dec); rimTT, et = val_H(rimH)
        pEt_ = []
        for pd,_d,_m,_n,_rn in zip((Et[1],et[1]), (D,d), (M,m), (n,en), (1,rn)):  # only d was projected
            pdd = pd - d * dec * _rn
            pm = _m * dec * _rn
            val_pdd = pdd * (pm / (ave*_n))  # val *= (_d/ad if fi else _m/am) * (mw*(_n/n))
            pEt_ += [np.array([pm-val_pdd, pd, _n])]
        Et, et = pEt_
        if val_(Et+et, aw=clust_w):
            ET+=Et; eT+=et; DerTT+=derTT; RimTT+=rimTT
            rim = CN(N_=_N.N_, Et=Et,et=et, derTT=derTT,extTT=rimTT, derH=derH,extH=rimH, root=CN())
            N_ += [CN(N_=_N.N_, Et=Et,et=et, derTT=derTT,extTT=rimTT, derH=derH,extH=rimH, root=CN())]  # same target position?
    # proj Fg:
    if val_(ET+eT, mw=len(N_)*Lw, aw=clust_w):
        return CN(N_=N_,L_=Fg.L_,Et=ET+eT, derTT=DerTT+RimTT)

def L_olp(C,_C, link_, frim=0):  # rim, link_ olp eval for centroids

    if link_: # R or C
        Lo_Et = np.sum([l.Et for l in list(set(link_) & set(_C.L_))]) * int_w
        LEt = _C.Et * int_w; rM = Lo_Et[0] / LEt[0]; Et = Lo_Et
    else:
        rM = 1; Et = np.zeros(3)
    if frim:  # N or C
        ro_Et = np.sum([l.Et for l in list(set(C.rim) & set(_C.rim))])
        rEt = _C.rim.Et; rM *= ro_Et[0]/rEt[0]; Et += ro_Et

    if val_(Et*rM, aw=clust_w) < 0:
        return 1  # continue

def rolp(N, link_=[], rim=[]):  # relative rim, link_ olp eval for clustering, replace rolp_M?

    LrM, RrM, LEt, REt = 0,0,[],[]
    if link_: # R | C
        oL_ = [L for L in N.L_ if L in link_]
        if oL_:
            oEt = np.sum([l.Et for l in oL_]) *int_w; LEt = N.Et *int_w; LrM = oEt[0]/LEt[0]
    if rim:  # N | C pairwise
        oR_ = [L for L,_ in N.rim.L_ if L in rim.L_]
        if oR_:
            oEt = np.sum([l.Et for l in oR_]); REt = N.rim.Et; RrM = oEt[0]/REt[0]
    _rM = 0
    for rM,Et in zip((LrM,LEt),(RrM,REt)):
        if rM:
            if _rM: _rM*= rM; _Et+= Et
            else:   _rM = rM; _Et = Et

    return val_(_Et*_rM, aw=clust_w) if _rM else 0


def rolp_M(M, N, _N_, fi, fC=0):  # rel sum of overlapping links Et, or _N_ Et in cluster_C_?

    if fC:  # from cluster_C_
        olp_N_ = [n for n in N.rim.N_ if n in _N_]  # previously accumulated inhibition zone
    else:   # from sel_exemplars
        olp_N_ = [L for L,_ in (N.rim.L_ if fi else N.rim.L_[0]+N.rim.L_[1]) if [n for n in L.N_ if n in _N_]]  # += in max dist?

    return 1 + sum([n.Et[0] for n in olp_N_]) / M  # w in range 1:2, rel_oM

def Cluster_C_(c_, rc, root):  # cluster centroids in c_ by rel_val of overlap in their members

    C_ = []  # flood-filled clusters of centroids
    for c in c_:
        if c.fin: continue
        C = CN(N_=c.N_, Et=copy(c.Et), olp=c.olp)  # init C cluster
        oN_ = [n for n in c.N_ if len(n.C_) > 1]  # each n has not-c centroids
        oc_ = set([_c for on in oN_ for _c in on.C_ if _c is not c])  # not-c centroids in all member N.C_s
        for N in oN_:
            for _c in N.C_:
                if _c is not c:
                    oEt = np.sum([n.Et for n in _c.N_ if n in oN_], axis=0)
                    if val_(oEt/_c.Et, aw=clust_w) > arn:  # element-wise
                       add_N(C,_c)
                       _c.fin=1
        C_ += [C]

    if val_(ET, mw=(len(C_)-1)*Lw, aw=rc+loop_w) > 0:
        root.C_ = C_  # higher-scope cross_comp in
        Ec_, Et = get_exemplars(root, [n for C in C_ for n in C.N_], rc+loop_w, fi=fi, fC=1)
        if Ec_ and val_(Et, mw=(len(Ec_)-1)*Lw, aw=rc+loop_w) > 0:
            # refine exemplars,
            # else keep C_, no further clustering?
            remove_ = {n for C in C_ for n in C.N_}
            E_[:] = [n for n in E_ if n not in remove_] + Ec_
            return (C_, ET+eEt)

def cluster_R_(_R_, rc):  # merge root centroids if L_overlap Et, pairwise

    R_ = []; Et = np.zeros(3)
    for R in _R_:
        if R.fin: continue  # merged
        # or in R.rim.N_ after cross_comp
        for _R in set([C for n in R.N_ for C in n.C_ if C is not R]):  # overlapping root centroids
            oN_ = list(set(R.N_) & set(R.N_))
            oV = val_(np.sum([n.Et for n in oN_], axis=0))
            aw = rc+clust_w
            _V = min(val_(_R.Et,aw), val_(R.Et,aw))
            if oV /_V > arn:  # pairwise eval, add link V if cross_comp
                add_N(R,_R, fmerge=1)
                _R.fin = 1
        R_ += [R]; Et += R.Et

    return CN(N_=R_,Et=Et)

def copy_(N, root=None, init=0):

    C = CN(root=root)
    N.root = C if init else root
    for name, val in N.__dict__.items():
        if name == "_id" or name == "root": continue
        elif name == "N_" and init: C.N_ = [N]
        elif name == "L_" and init: C.L_ = []
        elif name == "H"  and init: C.H = []
        elif name == "rim":  C.rim = copy_(val) if val else []  # rim of rim
        elif name == 'derH': C.derH = [[fork.copy_() if fork else [] for fork in lay] for lay in N.derH]
        elif isinstance(val,list) or isinstance(val,np.ndarray):
            setattr(C, name, copy(val))  # Et,yx,box, node_,link_,rim, alt_, baseT, derTT
        else:
            setattr(C, name, val)  # numeric or object: fi, root, span, nnest, lnest
    return C

    if G_:
        for G in G_:
            if val_(G.Et, (len(G.N_)-1)*Lw, rc+1, fi) > 0:
                # divisive sub-clustering with reduced rc, redundant to cluster_C_?
                G.N_ = cluster_N_(G.N_, rc* (1/ val_(root.Et, (len(root.N_)-1)*Lw)), fi=fi, root=G, _Nt=G).N_
        _Nt = CN(N_=G_,Et=Et)

    C._N_ = list(set([_n for n in _C._N_ for _n in n.rim.N_]))  # extend surround, add olp?

    return _Nt   # root N_|L_ replacement

def rn_olp(N, inhib_, fC):  # inhibition zone for centroids & exemplars

    if fC: olp_ = [n for n in N.rim.N_ if n in inhib_]
    else:  olp_ = [L for L,_ in N.rim.L_ if [n for n in L.N_ if n in inhib_]]  # += in max dist?
    if olp_:
        oEt = np.sum([i.Et for i in olp_], axis=0)
        rM = oEt[0] / N.Et[0]
        return val_(N.Et*rM, clust_w)
    else: return 0

def comb_alt_(G_, rc=1): # combine contour G.altG_ into altG (node_ defined by root=G),

    for G in G_:  # internal vs. external alts: different decay / distance, background + contour?
        if G.alt:
            if isinstance(G.alt, list):
                G.alt = sum_N_(G.alt,root=G)
                if val_(G.alt.Et, aw=G.olp, _Et=G.Et, fi=0):  # alt D * G rM
                    if not G.alt.N_[0].fi: L2N(G.alt.N_)
                    cross_comp(G.alt.N_, rc,fi=G.alt.N_[0].fi, root=G.alt)  # adds nesting
        elif G.H:  # not PP, sum dlinks:
            dL_ = list(set([L for g in G.N_ for L,_ in get_rim(g) if val_(L.Et, aw=G.olp, _Et=G.Et, fi=0) > 0]))
            if dL_ and val_(np.sum([l.Et for l in dL_],axis=0), aw=10+G.olp, _Et=G.Et, fi=0) > 0:
                G.alt = sum_N_(dL_,root=G)

def sum_N_(node_, root=None):  # form cluster G

    G = copy_(node_[0], root, init=1)
    for n in node_[1:]: add_N(G,n)
    if G.alt: # list
        G.alt = sum_N_(G.alt)  # internal
        if val_(G.alt.Et, aw=G.olp, _Et=G.Et, fi=0):
            cross_comp(G.alt.N_, G.olp+clust_w, root=G.alt)
    G.olp /= len(node_)
    for L in G.L_:
        if L not in G.L_: G.Et += L.Et; G.L_+=[L]
    # no rim
    return G

def add_N(N,n, fmerge=0, fC=0, rev=[]):

    if fmerge:
        for node in n.N_: node.root=N; N.N_ += [node]
        N.L_ += n.L_  # no L.root assign
    elif fC and n.rim:
        add_N(N.rim, n.rim)
    elif rev:  # N=rim, n=link
        rev,_N = rev
        N.N_ += [_N]
        if N.fi: N.L_ += [(n,rev)]  # N.L_ is rim links
        else:    N.L_[1-rev] += [(n,rev)]  # N.L_ is rimt, opposite to _N,N dir
    else:
        n.root=N; N.N_ += [n]
        if n.rim: N.L_ += n.rim.L_ if N.fi else n.rim.N_  # rim else
    if n.alt: N.alt += [n.alt]  # always flat?
    if n.H: add_NH(N.H, n.H, root=N)
    if n.lH: add_NH(N.lH, n.lH, root=N)
    rn = n.Et[2]/N.Et[2]
    if n.derH: add_H(N.derH,n.derH, N, rn)
    for Par,par in zip((N.angle, N.baseT, N.derTT), (n.angle, n.baseT, n.derTT)):
        Par += par * rn
    N.Et += n.Et * rn
    N.olp = (N.olp + n.olp * rn) / 2  # ave?
    N.yx = (N.yx + n.yx * rn) / 2
    N.span = max(N.span,n.span)
    N.box = extend_box(N.box, n.box)
    if hasattr(n,'C_') and hasattr(N,'C_'):
        N.C_ += n.C_; N.vo_ += n.vo_
    return N

def cross_comp1(iN_, rc, root, fi=1):  # rc: redundancy+olp; (cross-comp, exemplar selection, clustering), recursion

    N__,L_,Et = comp_node_(iN_,rc) if fi else comp_link_(iN_,rc)  # CLs if not fi, olp comp_node_(C_)?
    if N__:
        Nt, n__, cL_,fC = [],[],[],0
        for n in {N for N_ in N__ for N in N_}: n__ += [n]; n.sel = 0  # for cluster_N_
        # mfork:
        if val_(Et, (len(n__)-1)*Lw, rc+loop_w) > 0:
            if fi:
                E_,cL_,eEt, fC = get_exemplars(root, n__, rc+loop_w, fi)  # focal nodes, ~ point cloud
                if fC:  # E_ = C_, no cluster_N_, or refine exemplars by centroid M?
                    if cL_: L_ = cL_  # not empty if from cluster_NC_
                else:
                    if E_ and val_(eEt, (len(E_)-1)*Lw, rc+clust_w) > 0:
                        for rng, N_ in enumerate(N__,start=1):  # bottom-up rng incr
                            rng_E_ = [n for n in N_ if n.sel]; rc+=clust_w*rng  # cluster via exemplars
                            if rng_E_ and val_(np.sum([n.Et for n in rng_E_], axis=0), (len(rng_E_)-1)*Lw, rc) > 0:
                                Nt = cluster_N_(rng_E_, rc, fi,rng, _Nt=Nt)  # top G_
            else:
                Nt = cluster_N_(n__, rc+clust_w, fi, rng=1, _Nt=Nt)  # cluster L_/ nodet
            if Nt:
                if val_(Nt.Et, (len(Nt.N_)-1)*Lw, rc+loop_w, _Et=Et) > 0:
                    L_ = cross_comp(Nt.N_, rc, root=Nt)  # top Nt agg+
            elif fC:
                Nt = sum_N_(E_)  # Nt=Ct, no L_=cL_, Et=eEt?
        # dfork
        root.L_ = L_  # links between top Ns | Cs
        derH = sum_H_([L.derH for L in L_], root)
        root.angle = np.sum([L.angle for L in L_], axis=0)
        dval = val_(Et, (len(L_)-1)*Lw, rc+3+loop_w, fi=0)
        if dval > 0:
            L_ = L2N(L_)
            if dval > ave:  # recursive derivation -> lH/nLev, rng-banded?
                LL_ = cross_comp(L_, rc+loop_w*2, root, fi=0)  # comp_link_, select exemplars, centroids?
                if LL_: append_dH(derH, sum_H_([L.derH for L in LL_], root))  # ddfork
            else:  # lower res
                Lt = cluster_N_(L_, rc+clust_w*2, fi=0, fL=1, root=root)
                if Lt: root.Et += Lt.Et; root.lH += [Lt] + Lt.H  # link graphs, flatten H if recursive?
        append_dH(root.derH, derH)
        if Nt:
            lev = CN(N_=Nt.N_, Et=Nt.Et)  # N_ is new top H level
            add_NH(root.H, Nt.H+[lev], root)  # same depth?
            if Nt.H:  # lower levs: derH,H if recursion
                root.N_ = Nt.H.pop().N_  # top lev nodes
            root = Nt # higher composition and derivation, including L_ and lH

    return L_  # may be LL_ or agg+ L_

def comp_H(H,h, rn, root, DerTT=None, ET=None):  # one-fork derH if not fi, else two-fork derH

    derH, derTT, Et = [], np.zeros((2,8)), np.zeros(3)
    for _lay,lay in zip_longest(H,h):  # different len if lay-selective comp?
        if _lay and lay:
            dlay = []
            for _fork, fork in zip_longest(_lay, lay):  # dfork: xproj in link | xlink in node
                if _fork and fork:
                    dfork = _fork.comp_lay(fork, rn, root=root)
                    if dfork: dlay += [dfork]; derTT = dfork.derTT; Et += dfork.Et
                else: dlay += [[]]
            derH += [dlay]
    if Et[2]: DerTT += derTT; ET += Et
    return derH

def add_H(H, h, root, rev=0, rn=1):  # add fork derHs

    for Lay, lay in zip_longest(H,h):  # different len if lay-selective comp
        if lay:
            if Lay:
                for F, f in zip_longest(Lay, lay):
                    if f:
                        if F: F.add_lay(f)
                        else: Lay[1] = f.copy_(rev)  # default mfork
            else:
                H += [[f.copy_(rev) if f else [] for f in lay]]
            for fork in lay:
                if fork: root.derTT += fork.derTT*rn; root.Et += fork.Et*rn
    return H

def sum_H_(Q, Lay=CLay()):  # sum derH in link_|node_
    for derH in Q:
        for lay in derH: Lay.add_lay(lay)  # merge in dfork?
    return Lay

def append_dH(H, dH):  # merged mfork += [merged dfork], keep nesting

    for lay, dlay in zip_longest(H,dH):  # different len if lay-selective comp
        if dlay:
            if dlay[1]: dlay[0].add_lay(dlay[1]); dlay[1] = []
        if lay:
            if lay[1]: lay[0].add_lay(lay[1])
            if dlay:   lay[1] = dlay[0].copy_()
        elif dlay:     H += [dlay]
    return H

def cluster_C_(root, E_, rc):  # form centroids from exemplar _N_, drifting / competing via rims of new member nodes

    def comp_C(C, n):
        _,et,_ = base_comp(C, n)
        if C.rim and n.rim: _,_et,_= base_comp(C.rim,n.rim); et +=_et
        if C.alt and n.alt: _,_et,_= base_comp(C.alt,n.alt); et +=_et
        return et
    _C_ = []
    for E in E_:
        Et = E.rim.Et; C = copy_(E,root); C.N_=[E]; C.L_=[]; C.Et=Et  # init centroid, reset some other attrs?
        C._N_ = list({_n for n in E.rim.N_ for _n in n.rim.N_})  # core members + surround for comp to N_ mean
        _C_ += [C]
    while True:
        C_, ET, Olp, Dvo = [],np.zeros(3),0,np.zeros(2)  # per _C_
        for _C in sorted(_C_, key=lambda n: n.rim.Et[0]/n.rim.Et[2], reverse=True):
            # olp via _N_ of higher prior-loop vals?
            if val_(_C.Et, (len(_C._N_)-1)*Lw, clust_w+rc) > 0:
                # get mean cluster:
                C = sum_N_(_C.N_, root=root, fC=1)  # merge rim,alt before cluster_NC_
                _N_,_N__, Et,O,dvo = [],[], np.zeros(3),0,np.zeros(2)  # per C
                for n in _C._N_:  # core + surround
                    if C in n.C_: continue
                    et = comp_C(C,n); v = val_(et,aw=rc)
                    if v > 0:
                        olp = np.sum([vo[0] / v for vo in n.vo_ if vo[0] > v])  # olp: rel n val in stronger Cs, not summed with C
                        vo = np.array([v,olp]); O += olp  # current Cs overlap: norm by n+o
                        if v > ave * olp:
                            n.C_ += [C]; n.vo_ += [vo]; Et += et; _N_ += [n]; _N__ += n.rim.N_
                        if C not in n._C_: dvo += vo
                    elif C in n._C_:
                        dvo += n._vo_[n._C_.index(C)]  # old vo_
                Dvo += dvo  # new incl or excl
                C.Et = Et; C.olp = O; C.N_ = list(set(_N_)); C._N_ = list(set(_N__))  # select core, surround
                if val_(C.Et, aw=clust_w+rc+ C.olp) > 0:
                    C_ += [C]; ET += Et; Olp += O
            else: break  # expensive
        if np.sum(Dvo) > ave * loop_w+clust_w:  # both V and O represent change, comeasurable?
            _C_ = C_
            for n in root.N_:
                n._C_ = n.C_; n._vo_= n.vo_; n.C_,n.vo_ = [],[]  # new n.C_s
        else: break
        # converged, also break if low ET: no expansion?
    V = val_(ET, aw=rc+loop_w + Olp)  # C_ value
    if V > 0:
        Nt = sum_N_(E_)
        if V * ((len(C_)-1)*Lw) > ave*clust_w:
            Nt = cluster_NC_(C_, rc+clust_w, Nt)
        return Nt
