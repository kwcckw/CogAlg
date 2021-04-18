'''
Cross-compare blobs with incremental adjacency, within a frame
'''

from class_cluster import ClusterStructure, NoneType

class CderBlob(ClusterStructure):

    _blob = object
    blob = object
    dI = int
    mI = int
    dA = int
    mA = int
    dG = int
    mG = int
    dM = int
    mM = int
    mP = int
    dP = int
    sub_derBlob = list

class CBblob(ClusterStructure):
    
    derBlob = object
    derBlob_ = list
    blob_ = list
    
    

ave_M = 100
ave_mP = 100
ave_dP = 100


def cross_comp_blobs(blob_):
    '''
    root function of comp_blob: cross compare blobs with their adjacent blobs in frame.blob_, including sub_layers
    '''
    
    derBlob__ = []
    for blob in blob_:

        checked_ids_ = [blob.id]; net_M = 0  # checked ids per blob: may be checked from multiple root blobs
        # between blobs:
        derBlob__ += comp_blob_recursive(blob, blob.adj_blobs[0], checked_ids_, net_M)


    bblob_ = form_bblob_(derBlob__)

    return bblob_

# draft, similar with derP_2_PP
def form_bblob_(derBlob__):
    '''
    form bblob with same sign derBlob
    '''
    
    bblob_ = []
    checked_ids = []

    for derBlob in derBlob__[1:]:
        if derBlob.id not in checked_ids: # current derBlob is not checked before
            checked_ids.append(derBlob.id)
            
            bblob = CBblob(derBlob=CderBlob()) # init new bblob
            accum_bblob(bblob,derBlob)         # accum derBlob into bblob
            
            if derBlob._blob.derBlob.id not in checked_ids: # current derBlob is not checked before
                form_bblob_recursive(bblob, derBlob, derBlob._blob.derBlob, checked_ids) # recursively add blob to bblob via the adjacency
           
            bblob_.append(bblob) # pack bblob after checking through all adjacents
        
    return(bblob_)
    
    
def form_bblob_recursive(bblob, _derBlob, derBlob, checked_ids):
    
        checked_ids.append(derBlob.id) 
        
        if (_derBlob.mP>0) == (derBlob.mP>0):        # same sign check for adjacent derBlob
            accum_bblob(bblob,derBlob._blob.derBlob) # accum same sign derBlob into bblob
            
            if derBlob._blob.derBlob.id not in checked_ids: # current derBlob is not checked before
                form_bblob_recursive(bblob, derBlob, derBlob._blob.derBlob, checked_ids)  # recursively add blob to bblob via the adjacency
         
        # no need to reinit bblob when sign changed, it will be initialized in the main loop
    
def accum_bblob(bblob, derBlob):
    
    # accumulate derBlob
    bblob.derBlob.accumulate(**{param:getattr(derBlob, param) for param in bblob.derBlob.numeric_params})
    # add derBlob and blob to list
    bblob.derBlob_.append(derBlob)
    bblob.blob_.append(derBlob.blob)
     

def comp_blob_recursive(blob, adj_blob_, checked_ids_, net_M):
    '''
    called by cross_comp_blob to recursively compare blob to adj_blobs in incremental layers of adjacency
    '''
    derBlob_ = []
    
    for adj_blob in adj_blob_:
        if adj_blob.id not in checked_ids_:

            derBlob = comp_blob(blob,adj_blob)          # cross compare blob and adjacent blob
            net_M += adj_blob.Dert.M          # cost for extending the cross comparison
            checked_ids_.append(adj_blob.id)
            derBlob_.append(derBlob) 
            
            if (derBlob.mP> ave_mP):
                for _sub_layer, sub_layer in zip(blob.sub_layers, adj_blob.sub_layers):
                    
                    if _sub_layer and sub_layer: # both not empty sub layer
                        derBlob.sub_derBlob += cross_comp_blobs(_sub_layer)  # sub_blobs, add more conditions?
                        derBlob.sub_derBlob += cross_comp_blobs(sub_layer)  # could be done per blob only, but we need to do it before comp layers
            
                        '''
                        if _sub_layer[0].fork == sub_layer[0].fork:  # add compare other layer-wide parameters?
                            _sub_layer[0].append([sub_layer[0]])  # merge layers: temporary, unmerge if low cross-sub_blobs match?
                            cross_comp_blobs(_sub_layer[0])  # between sub blobs of same-depth layers of _blob and blob, skip same-blob cross-comp
                            # we need additional conditions in cross_comp_blobs for that?
                        '''    
            # search adjacents of adjacent if crit, depth-first
            if blob.Dert.M - net_M > ave_M:
                derBlob_ += comp_blob_recursive(blob, adj_blob.adj_blobs[0], checked_ids_, net_M)

            if blob.fsliced and adj_blob.fsliced:
                # apply comp_PP here to PPmm, Ppdm, PPmd, PPdd
                pass


    return derBlob_

def comp_blob(_blob, blob):
    '''
    cross compare _blob and blob
    '''
    (_I, _Dy, _Dx, _G, _M, _Dyy, _Dyx, _Dxy, _Dxx, _Ga, _Ma, _Mdx, _Ddx), _A = _blob.Dert.unpack(), _blob.A
    (I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, Mdx, Ddx), A = blob.Dert.unpack(), blob.A

    dI = _I - I  # d is always signed
    mI = min(_I, I)
    dA = _A - A
    mA = min(_A, A)
    dG = _G - G
    mG = min(_G, G)
    dM = _M - M
    mM = min(_M, M)
    mP = mI + mA + mG + mM
    dP = dI + dA + dG + dM

    # form derBlob regardless 
    derBlob  = CderBlob(_blob=_blob, blob=blob, dI=dI, mI=mI, dA=dA, mA=mA, dG=dG, mG=mG, dM=dM, mM=mM, mP=mP, dP=dP)
    blob.derBlob = derBlob

    return derBlob
            

