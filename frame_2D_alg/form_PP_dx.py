'''
Descriptions here

'''

aveG = 100

# still a draft, not quite sure and not completed yet
def form_PP_dx_(PP_):  # convert selected PP into gPP, should be run over the whole PP_ ?

    ave_PP = 100  # min summed value of gdert params
    for PP in PP_:

        # get unique Pys in PP
        Py_ = set([dderP.P for dderP in PP.derP_] + [dderP._P for dderP in PP.derP_])
        
        gPPy_ = []  
        
        PP_Dg = PP.derPP.dDg
        PP_Mg = PP.derPP.mDg
        

        if PP.derPP.dDg > aveG:
            Py_, Dg, Mg = comp_g(Py_)  # adds gdert_, Dg, Mg per P
            PP_Dg += abs(Dg) 
            PP_Mg += Mg
            
        if PP_Dg + PP_Mg < ave_PP:  #  revert to Py_ if below-cost
            # terminate PP
            gPPy_.append(gPP)  # pack gPP, should we create new class for gPP? Or reuse PP class?
            PP.gPPy_ = gPPy_
        else:
            PP.gPPy = Py_


def comp_g(Py_):  # cross-comp of gs in P.dert_, in gPP.Py_
    gP_ = []
    gP_Dg = gP_Mg = 0

    for P in Py_:
        Dg=Mg=0
        gdert_ = []
        _g = P.dert_[0][3]  # first g
        for dert in P.dert_[1:]:
            g = dert[3]
            dg = g - _g
            mg = min(g, _g)
            gdert_.append((dg, mg))  # no g: already in dert_
            Dg+=dg  # P-wide cross-sign, P.L is too short to form sub_Ps
            Mg+=mg
            _g = g
        P.gdert_ = gdert_
        P.Dg = Dg
        P.Mg = Mg
        gP_.append(P)
        gP_Dg += Dg
        gP_Mg += Mg  # positive, for stack evaluation to set fPP

    return gP_, gP_Dg, gP_Mg


def form_gP_(gdert_):  # probably not needed.
    gP_ = []
    _g, _Dg, _Mg = gdert_[0]  # first gdert
    _s = _Mg > 0  # initial sign

    for (g, Dg, Mg) in gdert_[1:]:
        s = Mg > 0  # current sign
        if _s != s:  # sign change
            gP_.append([_s, _Dg, _Mg])  # pack gP
            # update params
            _s = s
            _Dg = Dg
            _Mg = Mg
        else:  # accumulate params
            _Dg += Dg  # should we abs the value here?
            _Mg += Mg

    gP_.append([_s, _Dg, _Mg])  # pack last gP
    return gP_