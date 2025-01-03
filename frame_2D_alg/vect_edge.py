from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_
from slice_edge import slice_edge, comp_angle, ave_G
from comp_slice import comp_slice, comp_latuple, comp_md_
from itertools import combinations, zip_longest
from copy import deepcopy, copy
import numpy as np

'''
This code is initially for clustering segments within edge: high-gradient blob, but too complex for that.
It's mostly a prototype for open-ended compositional recursion: clustering blobs, graphs of blobs, etc.
-
rng+ fork: incremental-range cross-comp nodes: edge segments at < max distance, cluster if they match. 
der+ fork: incremental-derivation cross-comp links, from node cross-comp, if abs_diff * rel_adjacent_match 
(variance patterns borrow value from co-projected match patterns because their projections cancel-out)
- 
So graphs should be assigned adjacent alt-fork (der+ to rng+) graphs, to which they lend predictive value.
But alt match patterns borrow already borrowed value, which is too tenuous to track, we use average borrowed value.
Clustering criterion within each fork is summed match of >ave vars (<ave vars are not compared and don't add comp costs).
-
Clustering is exclusive per fork,ave, with fork selected per variable | derLay | aggLay 
Fuzzy clustering can only be centroid-based, because overlapping connectivity-based clusters will merge.
Param clustering if MM, compared along derivation sequence, or combinatorial?
-
Summed-graph representation is nested in a dual tree of down-forking elements: node_, and up-forking clusters: root_.
That resembles a neuron, which has dendritic tree as input and axonal tree as output. 
But we have recursively nested param sets packed in each level of the trees, which don't exist in neurons.
-
diagrams: 
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/agg_recursion_unfolded.drawio.png
-
notation:
prefix  f denotes flag
postfix t denotes tuple, multiple ts is a nested tuple
prefix  _ denotes prior of two same-name variables, multiple _s for relative precedence
postfix _ denotes array of same-name elements, multiple _s is nested array
capitalized variables are usually summed small-case variables
'''
ave = 3
ave_d = 4
ave_L = 4
max_dist = 2
ave_rn = 1000  # max scope disparity
ccoef = 10  # scaling match ave to clustering ave
icoef = .15  # internal M proj_val / external M proj_val
med_cost = 10

class CH(CBase):  # generic derivation hierarchy of variable nesting: extH | derH, their layers and sub-layers

    name = "H"
    def __init__(He, Et=None, tft=None, lft=None, H=None, fd_=None, root=None, node_=None, altH=None):
        super().__init__()
        He.Et = np.zeros(4) if Et is None else Et
        He.tft = [] if tft is None else tft  # top fork tuple: arrays m_t, d_t
        He.lft = [] if lft is None else lft  # lower fork tuple: CH /comp N_,L_, each has tft and lft
        He.H   = [] if H is None else H
        He.fd_ = [] if fd_ is None else fd_  # 0: sum CGs, 1: sum CLs, + concat from comparands
        He.root = None if root is None else root  # N or higher-composition He
        He.node_ = [] if node_ is None else node_  # concat bottom nesting order if CG, may be redundant to G.node_
        He.altH = CH(altH=object) if altH is None else altH   # summed altLays, prevent cyclic
        # He.H = []  # combined layers of fork tree?
        # then:
        # He.i = 0 if i is None else i  # lay index in root.lft, to revise olp
        # He.i_ = [] if i_ is None else i_  # priority indices to compare node H by m | link H by d
        # He.fd = 0 if fd is None else fd  # 0: sum CGs, 1: sum CLs
        # He.ni = 0  # exemplar in node_, trace in both directions?
        # He.deep = 0 if deep is None else deep  # nesting in root H
        # He.nest = 0 if nest is None else nest  # nesting in H
    def __bool__(H): return bool(H.tft)  # empty CH


    def copy_(He, root=None, rev=0, fc=0, i=None):  # comp direction may be reversed to -1

        if i:  # reuse self
            C = He; He = i; C.lft = []; C.tft=[]; C.root=root; C.node_=copy(i.node_)
        else:  # init new C
            C = CH(root=root, node_=copy(He.node_), fd_=copy(He.fd_))
        C.Et = He.Et * -1 if (fc and rev) else copy(He.Et)

        for fd, tt in enumerate(He.tft):  # nested array tuples
            C.tft += [tt * -1 if rev and (fd or fc) else deepcopy(tt)]
        for fork in He.lft:  # empty in bottom layer
            C.lft += [fork.copy_(root=C, rev=rev, fc=fc)]

        if not i: return C

    def add_tree(HE, He_, root=None, rev=0, fc=0):  # rev = dir==-1, unpack derH trees down to numericals and sum/subtract them
        if not isinstance(He_,list): He_ = [He_]

        for He in He_:
            if HE:  # sum tft:
                for fd, (F_t, f_t) in enumerate(zip(HE.tft, He.tft)):  # m_t and d_t
                    for F_,f_ in zip(F_t, f_t):
                        F_ += f_ * -1 if rev and (fd or fc) else f_  # m_| d_ in [dext,dlat,dver]

                for F, f in zip_longest(HE.lft, He.lft, fillvalue=None):  # CH forks
                    if f:  # not bottom layer
                        if F: F.add_tree(f,rev,fc)  # unpack both forks
                        else:
                            f.root=HE; HE.lft += [f]; HE.Et += f.Et

                HE.node_ += [node for node in He.node_ if node not in HE.node_]  # empty in CL derH?
                HE.Et += He.Et * -1 if rev and fc else He.Et
                
                # sum H layers
                for H, h in zip_longest(HE.H, He.H, fillvalue=None):
                    if h is not None:
                        if H is not None:  
                            H.add_tree(h)
                        else:              
                            HE.H += [h.copy_()]
            else:
                HE.copy_(root,rev,fc, i=He)
        return HE

    def comp_tree(_He, He, rn, root, dir=1):  # unpack derH trees down to numericals and compare them

        _d_t, d_t = _He.tft[1], He.tft[1]  # comp_tft:
        d_t = d_t * rn  # norm by accum span
        dd_t = (_d_t - d_t * dir)  # np.arrays
        md_t = np.array([np.minimum(_d_,d_) for _d_,d_ in zip(_d_t, d_t)], dtype=object)
        for i, (_d_,d_) in enumerate(zip(_d_t, d_t)):
            md_t[i][(_d_<0) != (d_<0)] *= -1  # negate if only one of _d_ or d_ is negative
        M = sum([sum(md_) for md_ in md_t])
        D = sum([sum(dd_) for dd_ in dd_t])
        n = .3 if len(d_t)==1 else 2.3  # n comp params / 6 (2/ext, 6/Lat, 6/Ver)

        derH = CH(root=root, tft = [np.array(md_t),np.array(dd_t)], Et=np.array([M,D,n, (_He.Et[3]+He.Et[3])/2]))

        for _fork, fork in zip(_He.lft, He.lft):  # comp shared layers
            if _fork and fork:  # same depth
                subH = _fork.comp_tree(fork, rn, root=derH )  # deeper unpack -> comp_md_t
                derH.lft += [subH]  # always fd=0 first
                derH.Et += subH.Et
        return derH

    def norm_(He, n):
        for f in He.tft:  # arrays
            f *= n
        for fork in He.lft:  # CHs
            fork.norm_(n)
            fork.Et *= n
        He.Et *= n

    # not used:
    def sort_H(He, fd):  # re-assign olp and form priority indices for comp_tree, if selective and aligned

        i_ = []  # priority indices
        for i, lay in enumerate(sorted(He.lft, key=lambda lay: lay.Et[fd], reverse=True)):
            di = lay.i - i  # lay index in H
            lay.olp += di  # derR- valR
            i_ += [lay.i]
        He.i_ = i_  # comp_tree priority indices: node/m | link/d
        if not fd:
            He.root.node_ = He.lft[i_[0]].node_
            # no He.node_ in CL?

class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster

    def __init__(G,  **kwargs):
        super().__init__()
        # inputs:
        G.fd = kwargs.get('fd', 0)  # 1 if cluster of Ls | lGs?
        G.Et = kwargs.get('Et', np.zeros(4))  # sum all param Ets
        G.root = kwargs.get('root')  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
        G.node_ = kwargs.get('node_', [])  # convert to GG_ or node_tree in agg++
        G.link_ = kwargs.get('link_', [])  # internal links per comp layer in rng+, convert to LG_ in agg++
        G.latuple = kwargs.get('latuple', np.array([.0,.0,.0,.0,.0,np.zeros(2)],dtype=object))  # lateral I,G,M,D,L,[Dy,Dx]
        G.vert = kwargs.get('vert', np.array([np.zeros(6), np.zeros(6)]))  # vertical m_d_ of latuple
        G.box = kwargs.get('box', np.array([np.inf,np.inf,-np.inf,-np.inf]))  # y0,x0,yn,xn
        G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = [(y0+yn)/2,(x0,xn)/2], then ave node yx
        G.rim = []  # flat links of any rng, may be nested in clustering
        G.maxL = kwargs.get('maxL', 0)  # nesting in nodes
        G.aRad = 0  # average distance between graph center and node center
        # maps to node_tree / agg+|sub+:
        G.derH = CH()  # sum from nodes, then append from feedback
        G.extH = CH()  # sum from rims
        G.altG_ = []  # or altG? adjacent (contour) gap+overlap alt-fork graphs, converted to CG
        # fd_ | fork_tree: list = z([[]])  # indices in all layers(forks, if no fback merge
        # G.fback_ = []  # fb buffer
    def __bool__(G): return bool(G.node_)  # never empty

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"

    def __init__(l, nodet, derH, yx, angle, dist, box):
        super().__init__()
        # CL = binary tree of Gs, depth+/der+: CL nodet is 2 Gs, CL + CLs in nodet is 4 Gs, etc., unpack sequentially
        l.derH = derH
        l.nodet = nodet  # e_ in kernels, else replaces _node,node: not used in kernels
        l.angle = angle  # dy,dx between nodet centers
        l.dist = dist  # distance between nodet centers
        l.box = box  # sum nodet, not needed?
        l.yx = yx
        # add med, rimt, elay | extH in der+
    def __bool__(l): return bool(l.nodet)

def vectorize_root(frame):
    # init for agg+:
    frame2CG(frame, derH=CH(root=frame, Et=np.zeros(4), tft=[]), root=None)  # distinct from base blob_
    blob_ = unpack_blob_(frame)
    for blob in blob_:
        if not blob.sign and blob.G > ave_G * blob.root.olp:
            edge = slice_edge(blob)
            if edge.G * (len(edge.P_)-1) > ave:  # eval PP
                comp_slice(edge)
                if edge.Et[0] * (len(edge.node_)-1)*(edge.rng+1) > ave:
                    lat = np.array([.0,.0,.0,.0,.0,np.zeros(2)],dtype=object); vert = np.array([np.zeros(6), np.zeros(6)])
                    for PP in edge.node_:
                        vert += PP[3]; lat += PP[4]
                    y_,x_ = zip(*edge.dert_.keys()); box = [min(y_),min(x_),max(y_),max(x_)]
                    blob2CG(edge, root=frame, vert=vert,latuple=lat, box=box, yx=np.divide([edge.latuple[:2]], edge.area))  # node_, Et stays the same
                    G_ = []
                    for N in edge.node_:  # no comp node_, link_ | PPd_ for now
                        P_, link_, vert, lat, A, S, box, [y,x], Et = N[1:]  # PPt
                        if Et[0] > ave:   # no altG until cross-comp
                            PP = CG(fd=0, Et=Et,root=edge, node_=P_,link_=link_, vert=vert, latuple=lat, box=box, yx=[y,x])
                            y0,x0,yn,xn = box; PP.aRad = np.hypot((yn-y0)/2,(xn-x0)/2)  # approx
                            G_ += [PP]
                    edge.node_ = G_
                    if len(G_) > ave_L:
                        cluster_edge(edge); frame.node_ += [edge]; frame.derH.add_tree(edge.derH)
                        # add altG: summed converted adj_blobs of converted edge blob
                        # if len(edge.node_) > ave_L: agg_recursion(edge)  # unlikely

def val_(Et, mEt=[], fo=0, coef=1):
    # ave *= coef: higher aves
    m, d, n, o = Et
    if any(mEt):
        mm,_,mn,_ = mEt  # cross-induction from root mG, not affected by overlap
        val = d * (mm / (ave * coef * mn)) - ave_d * coef * n * (o if fo else 1)
    else:
        val = m - ave * coef * n * (o if fo else 1)  # * overlap in cluster eval, not comp eval
    return val

def cluster_edge(edge):  # edge is CG but not a connectivity cluster, just a set of clusters in >ave G blob, unpack by default?

    def cluster_PP_(fd):
        G_ = []
        N_ = copy(edge.link_ if fd else edge.node_)
        while N_:  # flood fill
            node_,link_, et = [],[], np.zeros(4)
            N = N_.pop(); _eN_ = [N]
            while _eN_:
                eN_ = []
                for eN in _eN_:  # rim-connected ext Ns
                    node_ += [eN]
                    for L,_ in get_rim(eN, fd):
                        if L not in link_:
                            for eN in L.nodet:  # eval by link.derH.Et + extH.Et * ccoef > ave?
                                if eN in N_:
                                    eN_ += [eN]; N_.remove(eN)  # merged
                            link_ += [L]; et += L.derH.Et
                _eN_ = {*eN_}
            if val_(et) > 0:
                G_ += [sum2graph(edge, [node_,link_,et], fd)]
            else:
                G_ += node_  # unpack weak Gts
        if fd: edge.link_ = G_
        else:  edge.node_ = G_  # vs init PP_
    # comp PP_:
    N_,L_,Et = comp_node_(edge.node_)
    if val_(Et, fo=1) > 0:  # cancel by borrowing d?
        mlay = CH().add_tree([L.derH for L in L_]); H=edge.derH; mlay.root=H; H.Et += mlay.Et; H.lft = [mlay]; mH = mlay.H; dH = []  # init with mfork
        if len(N_) > ave_L:
            cluster_PP_(fd=0)
        if val_(Et, mEt=Et, fo=1) > 0:  # likely not from the same links
            for L in L_:
                L.extH, L.root, L.mL_t, L.rimt, L.aRad, L.visited_, L.Et = CH(), edge, [[],[]], [[],[]], 0, [L], copy(L.derH.Et)
            # comp dPP_:
            lN_,lL_,dEt = comp_link_(L_, Et)
            if val_(dEt, fo=1) > 0:
                dlay = CH().add_tree([L.derH for L in lL_]); dlay.root=H; H.Et += dlay.Et; H.lft += [dlay]; dH = dlay.H  # append dfork
                if len(lN_) > ave_L:
                    cluster_PP_(fd=1)

        # tentative:
        # merge H from both mlay and dlay before concatenate it into H
        HH = []
        for mh, dh in zip_longest(mH, dH, fillvalue=None):
            if not mh:   HH += [dh]
            elif not dh: HH += [mh]
            else:        HH += mh.copy_().add_tree(dh)
        H.H += HH
                            

def comp_node_(_N_):  # rng+ forms layer of rim and extH per N, appends N_,L_,Et, ~ graph CNN without backprop

    _Gp_ = []  # [G pair + co-positionals]
    for _G, G in combinations(_N_, r=2):
        rn = _G.Et[2] / G.Et[2]
        if rn > ave_rn: continue  # scope disparity
        radii = G.aRad + _G.aRad
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        _G.add, G.add = 0, 0
        _Gp_ += [(_G,G, rn, dy,dx, radii, dist)]
    rng = 1
    N_,L_,ET = set(),[],np.zeros(4)
    _Gp_ = sorted(_Gp_, key=lambda x: x[-1])  # sort by dist, shortest pairs first
    while True:  # prior vM
        Gp_,Et = [],np.zeros(4)
        for Gp in _Gp_:
            _G,G, rn, dy,dx, radii, dist = Gp
            _nrim = {L.nodet[1] if L.nodet[0] is _G else L.nodet[0] for L,_ in _G.rim}
            nrim = {L.nodet[1] if L.nodet[0] is G else L.nodet[0] for L,_ in G.rim}
            if _nrim & nrim:  # indirectly connected Gs,
                continue     # no direct match priority?
            # dist vs. radii * induction:
            if dist < max_dist * ((radii * icoef**3) * (val_(_G.Et)+val_(G.Et) + val_(_G.extH.Et)+val_(G.extH.Et))):
                Link = comp_N(_G,G, rn, angle=[dy,dx],dist=dist)
                L_ += [Link]  # include -ve links
                if val_(Link.derH.Et) > 0:
                    N_.update({_G,G}); Et += Link.derH.Et; _G.add,G.add = 1,1
            else:
                Gp_ += [Gp]  # re-evaluate not-compared pairs with one incremented N.M
        ET += Et
        if val_(Et) > 0:  # current-rng vM, -= borrowing d?
            _Gp_ = [Gp for Gp in Gp_ if Gp[0].add or Gp[1].add]  # one incremented N.M
            rng += 1
        else:  # low projected rng+ vM
            break

    return  list(N_), L_, ET  # flat N__ and L__

def comp_link_(iL_, iEt):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims

    fd = isinstance(iL_[0].nodet[0], CL)
    for L in iL_:
        # init mL_t: bilateral mediated Ls per L:
        for rev, N, mL_ in zip((0,1), L.nodet, L.mL_t):
            for _L,_rev in N.rimt[0]+N.rimt[1] if fd else N.rim:
                if _L is not L:
                    if val_(_L.derH.Et, mEt=iEt) > 0: # proj val = compared d * rel root M
                        mL_ += [(_L, rev ^_rev)]  # direction of L relative to _L
    _L_, out_L_, LL_, ET = iL_,set(),[],np.zeros(4)  # out_L_: positive subset of iL_, Et = np.zeros(4)?
    med = 1
    while True:  # xcomp _L_
        L_, Et = set(), np.zeros(4)
        for L in _L_:
            for mL_ in L.mL_t:
                for _L, rev in mL_:  # rev is relative to L
                    # if _L not in iL_, skip it? Or it shouldn't happen?
                    rn = _L.Et[2] / L.Et[2]
                    if rn > ave_rn: continue  # scope disparity
                    dy,dx = np.subtract(_L.yx,L.yx)
                    Link = comp_N(_L,L, rn,angle=[dy,dx],dist=np.hypot(dy,dx), dir = -1 if rev else 1)  # d = -d if L is reversed relative to _L
                    Link.med = med
                    LL_ += [Link]  # include -ves, link order: nodet < L < rimt, mN.rim || L
                    if val_(Link.derH.Et) > 0:  # link induction
                        out_L_.update({_L,L}); L_.update({_L,L}); Et += Link.derH.Et
        ET += Et
        if not any(L_): break
        # extend mL_t per last medL:
        if val_(Et) - med * med_cost > 0:  # project prior-loop value, med adds fixed cost
            _L_, ext_Et = set(), np.zeros(4)
            for L in L_:
                mL_t, lEt = [set(),set()], np.zeros(4)  # __Ls per L
                for mL_,_mL_ in zip(mL_t, L.mL_t):
                    for _L, rev in _mL_:
                        for _rev, N in zip((0,1), _L.nodet):
                            rim = N.rimt if fd else N.rim
                            if len(rim) == med:  # append in comp loop
                                for __L,__rev in rim[0]+rim[1] if fd else rim:
                                    if __L in L.visited_ or __L not in iL_: continue
                                    L.visited_ += [__L]; __L.visited_ += [L]
                                    if val_(__L.derH.Et, mEt=Et) > 0:  # compared __L.derH mag * loop induction
                                        mL_.add((__L, rev ^_rev ^__rev))  # combine reversals: 2 * 2 mLs, but 1st 2 are pre-combined
                                        lEt += __L.derH.Et
                if val_(lEt) > 0: # L'rng+, vs L'comp above
                    L.mL_t = mL_t; _L_.add(L); ext_Et += lEt
            # refine eval by extension D:
            if val_(ext_Et, mEt=Et) - med * med_cost > 0:
                med +=1
            else: break
        else: break
    return out_L_, LL_, ET

def extend_box(_box, box):  # extend box with another box
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return min(y0, _y0), min(x0, _x0), max(yn, _yn), max(xn, _xn)

def comp_area(_box, box):
    _y0,_x0,_yn,_xn =_box; _A = (_yn - _y0) * (_xn - _x0)
    y0, x0, yn, xn = box;   A = (yn - y0) * (xn - x0)
    return _A-A, min(_A,A) - ave_L**2  # mA, dA

def comp_N(_N,N, rn, angle=None, dist=None, dir=1):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = isinstance(N,CL)  # compare links, relative N direction = 1|-1
    # comp externals:
    if fd:
        _L, L = _N.dist, N.dist;  dL = _L - L*rn; mL = min(_L, L*rn) - ave_L  # direct match
        mA,dA = comp_angle(_N.angle, [d*dir *rn for d in N.angle])  # rev 2nd link in llink
        # comp med if LL: isinstance(>nodet[0],CL), higher-order version of comp dist?
    else:
        _L, L = len(_N.node_),len(N.node_); dL = _L- L*rn; mL = min(_L, L*rn) - ave_L
        mA,dA = comp_area(_N.box, N.box)  # compare area in CG vs angle in CL
    # der ext
    m_t = np.array([mL,mA],dtype=float); d_t = np.array([dL,dA],dtype=float)
    _o,o = _N.Et[3],N.Et[3]; olp = (_o+o) / 2  # inherit from comparands?
    Et = np.array([mL+mA, abs(dL)+abs(dA), .3, olp])  # n = compared vars / 6
    if fd:  # CH
        m_t = np.array([m_t]); d_t = np.array([d_t])  # add nesting
    else:   # CG
        (mLat, dLat), et1 = comp_latuple(_N.latuple, N.latuple, _o,o)
        (mVer, dVer), et2 = comp_md_(_N.vert[1], N.vert[1], dir)
        m_t = np.array([m_t, mLat, mVer], dtype=object)
        d_t = np.array([d_t, dLat, dVer], dtype=object)
        Et += np.array([et1[0]+et2[0], et1[1]+et2[1], 2, 0])
        # same olp?
    lay = CH(fd_=[fd], tft=[m_t,d_t], Et=Et)
    if _N.derH and N.derH:
        derH = _N.derH.comp_tree(N.derH, rn, root=lay)  # comp shared layers
        lay.Et += derH.Et; lay.lft = [derH]; lay.H += [derH.copy_()]  # CH.H[0] is always sum of current CH.lft?
    # spec: comp_node_(node_|link_), combinatorial, node_ nested / rng-)agg+?
    Et = copy(lay.Et)
    if not fd and _N.altG_ and N.altG_:  # not for CL, eval M?
        # altG_ was converted to altG
        alt_Link = comp_N(_N.altG_, N.altG_, _N.altG_.Et[2] / N.altG_.Et[2])
        lay.altH = alt_Link.derH
        Et += lay.altH.Et

    Link = CL(nodet=[_N,N], derH=lay, yx=np.add(_N.yx,N.yx)/2, angle=angle,dist=dist,box=extend_box(N.box,_N.box))
    lay.root = Link
    if val_(Et) > 0:
        for rev, node in zip((0,1), (N,_N)):  # reverse Link direction for _N
            if fd: node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            else:  node.rim += [(Link,dir)]
            node.extH.add_tree(Link.derH)
            node.Et += Et
    return Link

def get_rim(N,fd): return N.rimt[0] + N.rimt[1] if fd else N.rim  # add nesting in cluster_N_?

def sum_G_(node_):
    G = CG()
    for n in node_:
        G.latuple += n.latuple; G.vert += n.vert; G.aRad += n.aRad; G.box = extend_box(G.box, n.box)
        if n.derH: G.derH.add_tree(n.derH, root=G)
        if n.extH: G.extH.add_tree(n.extH)
    return G

def sum2graph(root, grapht, fd, minL=0, maxL=None, nest=0):  # sum node and link params into graph, aggH in agg+ or player in sub+

    node_, link_, Et = grapht
    graph = CG(fd=fd, Et=Et*icoef, root=root, link_=link_, minL=minL, maxL=maxL, nest=nest+1)  # nest is not needed? Probably not now, but why we need minL in G now?
    # arg Et is weaker if internal, maxL,minL: max and min L.dist in graph.link_
    yx = np.array([0,0]); yx_ = []
    derH = CH(root=graph)
    N_ = []
    for N in node_:
        if minL:  #>0, inclusive, = lower-layer exclusive maxL if G is distance-nested in cluster_N_,
            while N.root.maxL and minL != N.root.maxL:  # higher graph maxL must be > 0, unless L.dist is 0  (to skip edge and frame)
                N = N.root  # cluster prior-dist graphs instead of nodes
        if N not in N_: N_ += [N]
        graph.box = extend_box(graph.box, N.box)  # pre-compute graph.area += N.area?
        yx = np.add(yx, N.yx)
        yx_ += [N.yx]
        if isinstance(node_[0],CG):
            graph.latuple += N.latuple; graph.vert += N.vert
        if N.derH:
            derH.add_tree(N.derH, graph)
        graph.Et += N.Et * icoef ** 2  # deeper, lower weight
        N.root = graph
    graph.node_ = N_  # nodes or roots, link_ is still current-dist links only?
    # sum link_ derH:
    derLay = CH().add_tree([link.derH for link in link_],root=graph)  # root added in copy_ within add_tree
    if derH:
        derLay.lft += [derH]; derLay.H += derH.H; derLay.Et += derH.Et  # concatenate H here
    graph.derH = derLay
    L = len(node_)
    yx = np.divide(yx,L)
    dy,dx = np.divide( np.sum([ np.abs(yx-_yx) for _yx in yx_], axis=0), L)
    graph.aRad = np.hypot(dy,dx)  # ave distance from graph center to node centers
    graph.yx = yx
    if fd:  # dgraph, no mGs / dG for now  # and val_(Et, mEt=root.Et) > 0:
        altG_ = []  # mGs overlapping dG
        for L in node_:
            for n in L.nodet:  # map root mG
                mG = n.root
                if mG not in altG_:
                    mG.altG_ += [graph]  # cross-comp|sum complete altG_ before next agg+ cross-comp
                    altG_ += [mG]
    feedback(graph)  # recursive root.derH.add_fork(graph.derH)
    return graph

def feedback(node):  # propagate node.derH to higher roots

    while node.root:
        root = node.root
        lowH = addH = root.derH
        add = 1
        # i think we miss out the packing of CH.fd_?
        for i, fd in enumerate(addH.fd_):  # unpack top-down, each fd was assigned by corresponding level of roots
            if len(addH.lft) > fd:
                addH = lowH; lowH = lowH.lft[fd]  # keep unpacking
            else:
                # draft, probably wrong:
                if lowH.lft: lowH.H[-1].sum_tree(addH)  # sum tree to same layer H
                else:        lowH.H += [addH.copy()]
                lowH.lft += [addH.copy_()]  # fork was empty, init with He
                add = 0; break
        if add:  # add in fork initialized by prior feedback, else append above
            lowH.add_tree(addH, root)
        node = root

def frame2CG(G, **kwargs):
    blob2CG(G, **kwargs)
    G.node_ = kwargs.get('node_', [])
    G.derH = kwargs.get('node_', CH())

def blob2CG(G, **kwargs):
    # node_, Et stays the same:
    G.fd = kwargs.get('fd', 0)  # 1 if cluster of Ls | lGs?
    G.root = kwargs.get('root')  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
    G.link_ = kwargs.get('link_', [])  # internal links per comp layer in rng+, convert to LG_ in agg++
    G.latuple = kwargs.get('latuple', np.array([.0,.0,.0,.0,.0,np.zeros(2)],dtype=object))  # lateral I,G,M,D,L,[Dy,Dx]
    G.vert = kwargs.get('vert', np.array([np.zeros(6), np.zeros(6)]))  # vertical m_d_ of latuple
    G.box = kwargs.get('box', np.array([np.inf,np.inf,-np.inf,-np.inf]))  # y0,x0,yn,xn
    G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = [(y0+yn)/2,(x0,xn)/2], then ave node yx
    G.rim = []  # flat links of any rng, may be nested in clustering
    G.maxL = 0  # nesting in nodes
    G.aRad = 0  # average distance between graph center and node center
    # maps to node_tree / agg+|sub+:
    G.derH = CH()  # sum from nodes, then append from feedback
    G.extH = CH()  # sum from rims
    G.altG_ = []  # or altG? adjacent (contour) gap+overlap alt-fork graphs, converted to CG
    if not hasattr(G, 'node_'): G.node_ = []  # add node_ in frame
    return G

if __name__ == "__main__":
    image_file = './images/raccoon_eye.jpeg'
    image = imread(image_file)
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vectorize_root(frame)