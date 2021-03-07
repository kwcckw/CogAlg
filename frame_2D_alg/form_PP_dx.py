'''
Descriptions here

'''

ave_Dx = 10
ave_PP_Dx = 100


def form_PP_dx_(P__): 

    # comp_PP_dx
    for P in P__: # start from root P
            
        PP_Dx_ = [] # contains list of PDx_
        Dx_ = [] # contains list of Dx
        PDx_ = [] # contains contiguous Ps and dx> ave_DX
        Dx = 0 # for Dx accumulation
        
        if P.downconnect_cnt == 0:
            
            if P.Dx > ave_Dx: # accumulate dx and ps
                Dx += P.Dx
                PDx_.append(P)
                
            if P.upconnect_: # recursively process upconnects
                comp_PP_dx(P.upconnect_, Dx, PDx_, PP_Dx_, Dx_)
                
                PP_Dx_.append(PDx_) # save to PP_Dx after scanning all upconnects
                Dx_.append(Dx)
                
            elif Dx != 0: # terminate Pdx and Dx if Dx != 0
                PP_Dx_.append(PDx_)
                Dx_.append(Dx)
          
    # comp_dx
    for i, (Dx,PDx_) in enumerate(zip(Dx_,PP_Dx_)):
        if Dx > ave_PP_Dx: 
            dxP_, dxP_Ddx, dxP_Mdx = comp_dx_(PDx_)
            # is there any further processing here before proceed to replace P_ with dxP_?
            
def comp_PP_dx(derP_, iDx, iPDx_, PP_Dx_, Dx_):
              
    Dx = iDx
    PDx_ = iPDx_
    
    for derP in derP_:
    
        P = derP._P
        
        if P.Dx > ave_Dx: # accumulate dx and Ps
            Dx += P.Dx
            PDx_.append(P)
        
        else: # terminate and reset Pdx and Dx
            PP_Dx_.append(PDx_)
            Dx_.append(Dx)
            Dx = 0
            PDx_ = []   
          
        if P.upconnect_: # recursively process upconnects
            comp_PP_dx(P.upconnect_, Dx, PDx_, PP_Dx_, Dx_)

        elif Dx != 0 and id(PDx_) != id(iPDx_): # terminate newly created PDx and Dx in current function call after processed their upconnects
            PP_Dx_.append(PDx_)
            Dx_.append(Dx)


def comp_dx_(P_):  # cross-comp of dx s in P.dert_
    dxP_ = []
    dxP_Ddx = 0
    dxP_Mdx = 0

    for P in P_:
        Ddx = 0
        Mdx = 0
        
        dxdert_ = []
        _dx = P.dert_[0][2]  # first dx
        for dert in P.dert_[1:]:
            dx = dert[2]
            ddx = dx - _dx
            mdx = min(dx, _dx)
            dxdert_.append((ddx, mdx))  # no dx: already in dert_
            Ddx+=ddx  # P-wide cross-sign, P.L is too short to form sub_Ps
            Mdx+=mdx
            _dx = dx
            
        P.dxdert_ = dxdert_
        P.Ddx = Ddx
        P.Mdx = Mdx
        
        dxP_.append(P)
        
        dxP_Ddx += Ddx
        dxP_Mdx += Mdx  

    return dxP_, dxP_Ddx, dxP_Mdx

